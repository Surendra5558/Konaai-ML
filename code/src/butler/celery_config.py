# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the Celery configuration settings."""
from datetime import timedelta

from kombu import Exchange
from kombu import Queue
from src.utils.global_config import GlobalSettings

QueueName = "MLQueue"
ExchangeName = "MLExchange"

############################################################################################################
# Celery exchange and queue configurations
############################################################################################################
consumer_timeout_hrs = GlobalSettings().broker.ConsumerTimeoutHrs
queue_exchange_config = {
    "task_queues": [
        Queue(
            name=QueueName,
            exchange=Exchange(name=ExchangeName),
            routing_key=f"task.{QueueName}",
            queue_arguments={
                "x-consumer-timeout": int(
                    timedelta(hours=consumer_timeout_hrs).total_seconds() * 1000
                ),  # Timeout in milliseconds
            },
            durable=True,
        )
    ],
    "task_default_queue": QueueName,
    "task_default_exchange": ExchangeName,
    "task_default_routing_key": f"task.{QueueName}",
}

############################################################################################################
# Celery serialization and content configurations
############################################################################################################
serialization_config = {
    "task_serializer": "json",  # This is to ensure that the task is serialized in JSON format
    "result_serializer": "json",  # This is to ensure that the result is serialized in JSON format
    "accept_content": [
        "json"
    ],  # This is to ensure that the content type accepted is JSON
}

############################################################################################################
# Timezone configurations
############################################################################################################
timezone_config = {
    "timezone": "UTC",
    "enable_utc": True,
}

############################################################################################################
# Broker configurations
############################################################################################################
heartbeat_seconds = GlobalSettings().broker.HeartbeatSeconds
broker_config = {
    "broker_connection_retry_on_startup": True,  # This is to ensure that the broker retries the connection on startup
    "broker_connection_retry": True,
    "broker_connection_max_retries": 100,
    "broker_transport_options": {
        "confirm_publish": True,  # This is to ensure that the broker confirms the message publish
        "max_retries": 5,  # This is to ensure that the broker retries 5 times before failing
        "interval_start": 0,  # This is to ensure that the retry interval starts from 0 seconds
        "interval_step": 10,  # This is to ensure that the retry interval increases by 2 seconds
        "interval_max": 300,  # This is to ensure that the maximum retry interval is 30 seconds
        "heartbeat": heartbeat_seconds,  # This is to ensure that the broker heartbeat is set to given seconds
    },
    "broker_connection_timeout": (
        consumer_timeout_hrs * 60 * 60
    ),  # This is to ensure that the broker connection timeout is given in seconds
    "broker_heartbeat": heartbeat_seconds,  # This is to ensure that the broker heartbeat is set to given seconds
    "broker_pool_limit": None,  # This is to ensure that the broker pool limit is disabled
}

############################################################################################################
# Worker configurations
############################################################################################################
worker_config = {
    "worker_cancel_long_running_tasks_on_connection_loss": True,  # This is to ensure that the worker cancels long running tasks on connection loss
    "worker_prefetch_multiplier": 1,  # This is to ensure that the worker prefetches only one message at a time
    "worker_disable_rate_limits": True,  # This is to ensure that the worker disables rate limits
    "worker_pool": "solo",  # This is to ensure that the worker uses threads for processing tasks. Only supported option for windows
    "worker_concurrency": 1,  # This is to ensure that the worker processes only one task at a time
}

############################################################################################################
# Task configurations
############################################################################################################
task_config = {
    "task_acks_late": True,  # This is to ensure that the task is acknowledged after it has been processed
    "task_acks_on_failure_or_timeout": True,  # This is to ensure that the task is acknowledged on failure or timeout
    "task_reject_on_worker_lost": True,  # This is to ensure that the task is rejected if the worker is lost
    "task_annotations": {
        "*": {
            "max_retries": 0,  # This is to ensure that the task is retried 0 times before failing
            "default_retry_delay": 300,  # This is to ensure that the default retry delay is 300 seconds
            "autoretry_for": (
                Exception,
            ),  # This is to ensure that the task is retried for exceptions of type Exception
            "retry_backoff": True,  # This is to ensure that the task uses exponential backoff for retries
            "retry_backoff_max": 3600,  # This is to ensure that the maximum backoff time is 3600 seconds
            "retry_jitter": True,  # This is to ensure that the task uses jitter for retries to avoid thundering herd problem where many tasks retry at the exact same time
        }
    },
    "task_publish_retry": True,  # This is to ensure that the task is retried by publishing a new message
    "task_publish_retry_policy": {
        "max_retries": 100,  # This is to ensure that the task is retried 1000 times before failing
        "interval_start": 0,  # This is to ensure that the retry interval starts from 0 seconds
        "interval_step": 10,  # This is to ensure that the retry interval increases by 2 seconds
        "interval_max": 300,  # This is to ensure that the maximum retry interval is 30 seconds
    },
    "task_track_started": True,  # This is to ensure that the task is tracked when it starts
    "task_soft_time_limit": int(
        timedelta(hours=36).total_seconds()
    ),  # This is to ensure that the task has a soft time limit
    "task_always_eager": False,  # This is to ensure that the task is not always eager
}

############################################################################################################
# Backend configurations
############################################################################################################
backend_config = {
    "result_backend_always_retry": True,  # This is to ensure that the result backend always retries
    "result_backend_max_retries": 5,  # This is to ensure that the result backend retries 3 times before failing
    "result_expires": 0,  # This is to ensure that the result never expires
}


celery_config = (
    queue_exchange_config
    | serialization_config
    | timezone_config
    | broker_config
    | worker_config
    | task_config
)
