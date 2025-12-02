# # Copyright (C) KonaAI - All Rights Reserved
"""This starts the celery app"""
import ssl
from typing import Optional

from celery import Celery
from celery.schedules import crontab
from src.butler.celery_broker import get_broker_url
from src.butler.celery_config import celery_config
from src.butler.celery_result_backend import create_tables
from src.butler.celery_result_backend import custom_schema
from src.butler.celery_result_backend import get_backend_url
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.butler.celery_config import QueueName, ExchangeName


def create_celery_app() -> Optional[Celery]:
    """
    Creates and configures a Celery application instance.
    """
    try:
        celery_instance = Celery(__name__)

        # create tables for result backend
        if GlobalSettings().workerdb.is_db_connected:
            if not create_tables():
                Status.FAILED(
                    "Failed to create tables for result backend. Check database connection."
                )
                return None
        else:
            Status.FAILED(
                "Master database connection failed. Check database connection."
            )
            return None

        # setup celery app
        celery_instance.conf.update(
            broker_url=get_broker_url(),
            result_backend=get_backend_url(),
            worker_hijack_root_logger=False,
        )
        celery_instance.conf.update(celery_config)
        celery_instance.conf.database_engine_options = {"echo": False}
        celery_instance.conf.database_table_schemas = {
            "task": custom_schema,
            "group": custom_schema,
        }
        celery_instance.conf.task_default_exchange = ExchangeName
        celery_instance.conf.task_default_queue = QueueName

        # ssl config
        if GlobalSettings().broker.SSL:
            celery_instance.conf.broker_use_ssl = {
                "ssl_version": ssl.PROTOCOL_TLS_CLIENT,
                "cert_reqs": ssl.CERT_NONE,
            }

        # Configure beat schedule
        celery_instance.conf.beat_schedule = {
            "Model Monitoring": {
                "task": "src.automl.tasks.model_monitoring",  # Use string reference instead of function
                # run every 15 days
                "schedule": crontab(minute=0, hour=0, day_of_month="*/15"),
            }
        }

        # Beat specific configuration
        celery_instance.conf.update(
            beat_max_loop_interval=60 * 5,  # in seconds
        )

        # import tasks
        celery_instance.conf.imports = [
            "src.automl.tasks",
        ]
        celery_instance.autodiscover_tasks()

        return celery_instance

    except BaseException as e:
        Status.FAILED(
            "Error setting up celery. Check global configuration.", error=str(e)
        )
        return None


# Create the celery instance
celery = create_celery_app() or Celery(__name__)
