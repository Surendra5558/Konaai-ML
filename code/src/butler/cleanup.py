# # Copyright (C) KonaAI - All Rights Reserved
"""This module completes the cleanup activities for the local drive.

Returns:
    _type_: None
"""
import os
import shutil
import contextlib
from datetime import datetime
from pathlib import Path

from src.utils.conf import Setup
from src.utils.status import Status


class CleanFiles:
    """
    CleanFiles provides static methods for cleaning up old files and directories.
    This class includes utilities to:
    - Remove files and folders older than a specified number of days from given directories.
    - Determine if a file or folder is older than a specified threshold.
    - Log failures during cleanup operations.
    Typical usage involves calling `old_files_cleanup()` to automatically clean up temporary and log directories based on predefined age limits.
    """

    @staticmethod
    def old_files_cleanup():
        """
        Cleans up old files from specified directories.

        This function performs cleanup tasks by deleting files older than a specified number of days
        from the temporary and log directories. Specifically, it deletes files older than 1 day from
        the temporary directory and files older than 90 days from the log directory. If an error occurs
        during the cleanup process, it logs the failure status with the error details.
        """
        try:
            # Clean temporary files older than 1 day
            CleanFiles.delete_content(1, Setup().temp_path)

            # Clean log files older than 90 days
            CleanFiles.delete_content(90, Setup().log_path)
        except BaseException as _e:
            Status.FAILED("Error while cleaning up files", error=str(_e))

    @staticmethod
    def delete_content(number_of_days: int, path: str):
        """
        Delete files and folders older than the specified number of days.

        Args:
            number_of_days (int): The number of days to consider for deletion.
            path (str): The path to the directory to be cleaned up.
        """
        try:
            path = Path(path)
            if path.exists():
                for root_folder, folders, files in os.walk(path):
                    for folder in folders:
                        folder_path = Path(root_folder, folder)
                        if CleanFiles.is_old(folder_path, number_of_days):
                            # Use onerror handler to attempt to fix permission issues
                            def _on_rm_error(func, path, exc_info):
                                # func is os.remove or os.rmdir; try to chmod then retry
                                with contextlib.suppress(Exception):
                                    os.chmod(path, 0o700)
                                try:
                                    if os.path.isdir(path):
                                        os.rmdir(path)
                                    else:
                                        os.remove(path)
                                except Exception:
                                    # If still failing, log and continue
                                    Status.FAILED(
                                        "Permission error while deleting path",
                                        path=path,
                                    )

                            try:
                                shutil.rmtree(folder_path, onerror=_on_rm_error)
                            except PermissionError as _pe:
                                # Last resort: try to chmod recursively and retry
                                try:
                                    for root, dirs, files in os.walk(folder_path):
                                        for d in dirs:
                                            with contextlib.suppress(Exception):
                                                os.chmod(os.path.join(root, d), 0o700)
                                        for f in files:
                                            with contextlib.suppress(Exception):
                                                os.chmod(os.path.join(root, f), 0o600)
                                    shutil.rmtree(folder_path, onerror=_on_rm_error)
                                except Exception as _e:
                                    Status.FAILED(
                                        "Error while deleting folder after chmod",
                                        path=str(folder_path),
                                        error=str(_e),
                                    )
                    for file in files:
                        file_path = Path(root_folder, file)
                        if CleanFiles.is_old(file_path, number_of_days):
                            try:
                                os.remove(file_path)
                            except PermissionError:
                                # try to make file and parent directory writable then remove
                                try:
                                    parent_dir = file_path.parent
                                    # make file writable if possible
                                    with contextlib.suppress(Exception):
                                        os.chmod(file_path, 0o600)
                                    # ensure parent directory is writable (needed to unlink)
                                    with contextlib.suppress(Exception):
                                        os.chmod(parent_dir, 0o700)
                                    os.remove(file_path)
                                except Exception as _e:
                                    Status.FAILED(
                                        "Error while removing file",
                                        path=str(file_path),
                                        parent=str(parent_dir) if 'parent_dir' in locals() else None,
                                        error=str(_e),
                                    )
        except BaseException as _e:
            Status.FAILED("Error while cleaning up files", error=str(_e))

    @staticmethod
    def is_old(file_path: Path, number_of_days: int):
        """
        Check if a file or folder is older than a specified number of days.

        Args:
            file_path (Path): The path to the file or folder.
            number_of_days (int): The number of days to compare against.

        Returns:
            bool: True if the file/folder is older than the specified number of days, False otherwise.
        """
        try:
            stat = file_path.stat()

            # Use creation time if available (macOS/Windows), else fallback to modification time (Linux/Docker)
            file_time = getattr(stat, "st_birthtime", stat.st_mtime)
            file_time = datetime.fromtimestamp(file_time)

            current_time = datetime.now()
            return (current_time - file_time).days > number_of_days

        except BaseException as _e:
            Status.FAILED(
                "Error while checking file age", file_path=file_path, error=str(_e)
            )
            return False
