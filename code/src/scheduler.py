# # Copyright (C) KonaAI - All Rights Reserved
"""This module starts the celery beat scheduler process"""
import sys

from src.butler.celery_app import celery
from src.utils.status import Status


def main():
    """
    Starts a Celery beat scheduler process.
    """
    # setup celery app
    celery_app = celery
    if celery_app is None:
        Status.FAILED("Celery scheduler app is not set up. Exiting.")
        sys.exit(1)

    # Start the beat scheduler
    celery_app.start(
        argv=[
            "beat",
            "--loglevel=info",
            "--max-interval",
            "300",  # 5 minutes in seconds
        ]
    )


if __name__ == "__main__":
    main()
