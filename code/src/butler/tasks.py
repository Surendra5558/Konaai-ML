# # Copyright (C) KonaAI - All Rights Reserved
"""This module runs housekeeping tasks"""
from src.butler.cleanup import CleanFiles
from src.worker import celery


@celery.task(name="File Cleanup Task")
def file_cleanup_task():
    """
    Task to clean up old files.

    This function initializes a CleanFiles object and calls its
    old_files_cleanup method to remove outdated files from the system.
    """
    CleanFiles.old_files_cleanup()


if __name__ == "__main__":
    task = file_cleanup_task.delay()
