# # Copyright (C) KonaAI - All Rights Reserved
"""This module starts the celery worker process"""
import platform
import sys

from src.butler.celery_app import celery
from src.utils.status import Status


def main():
    """
    Starts a Celery worker process with appropriate pool type based on the operating system.
    """
    pool_type = "solo" if platform.system() == "Windows" else "threads"

    # setup celery app
    celery_app = celery
    if celery_app is None:
        Status.FAILED("Celery app is not set up. Exiting.")
        sys.exit(1)

    celery_app.worker_main(
        argv=[
            "worker",
            "--loglevel=info",
            "--pool",
            pool_type,
            "--without-heartbeat",
        ]
    )


if __name__ == "__main__":
    main()
