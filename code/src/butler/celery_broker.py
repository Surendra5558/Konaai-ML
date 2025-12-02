# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the Celery broker configuration settings."""
import ast
import json
import logging
from urllib.parse import quote

import pika.spec
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import BasicProperties
from src.butler.celery_config import QueueName
from src.utils.global_config import GlobalSettings
from src.utils.status import Status

# change pika logging level to ERROR
logging.getLogger("pika").setLevel(logging.ERROR)


class Message:
    """
    A class to represent a message with its properties and method frame.
    Attributes:
    ----------
    id : str
        The unique identifier of the message.
    body : dict
        The body of the message.
    properties : BasicProperties
        The properties of the message, including headers and timestamp.
    method_frame : pika.spec.Basic.GetOk
        The method frame of the message.
    Methods:
    -------
    to_dict() -> dict:
        Converts the message attributes to a dictionary format.
    """

    id: str
    body: dict
    properties: BasicProperties
    method_frame: pika.spec.Basic.GetOk

    def to_dict(self) -> dict:
        """
        Converts the task properties to a dictionary representation.

        Returns:
            dict: A dictionary containing the task details with the following keys:
                - "Task ID": The ID of the task.
                - "Task Name": The name of the task.
                - "Submission Time" (optional): The timestamp when the task was submitted.
                - "Task Args" (optional): The arguments of the task if available.
                - "Task Kwargs" (optional): The keyword arguments of the task if available.
        """
        d = {
            "Task ID": self.id,
            "Task Name": self.properties.headers.get("task"),
        }

        try:
            if self.properties.timestamp:
                d["Submission Time"] = self.properties.timestamp

            args = self.properties.headers.get("argsrepr")
            if args and args != "()":
                d["Task Args"] = args

            kwargs = self.properties.headers.get("kwargsrepr")
            if kwargs and kwargs != "{}":
                d["Task Kwargs"] = ast.literal_eval(kwargs)
        except BaseException as _e:
            pass

        return d


def get_pending_messages() -> list:
    """
    Fetches the list of pending messages from RabbitMQ.
    This function connects to the RabbitMQ server using the connection parameters
    specified in the global settings, retrieves the pending messages, and returns
    them as a list. If an error occurs during the process, it logs the error and
    returns an empty list.

    Returns:
        list: A list of pending messages from RabbitMQ. If an error occurs, an empty
        list is returned.
    """
    Status.INFO("Fetching pending messages from RabbitMQ.")
    connection = None
    channel = None

    try:
        # Connect to RabbitMQ
        # Load broker settings
        credentials = pika.PlainCredentials(
            username=GlobalSettings().broker.UserName,
            password=GlobalSettings().broker.Password.get_secret_value(),
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=GlobalSettings().broker.HostName,
                port=GlobalSettings().broker.Port,
                virtual_host=GlobalSettings().broker.VirtualHost,
                credentials=credentials,
            )
        )
        channel = connection.channel()
        return _list_pending_messages(channel)
    except BaseException as _e:
        Status.FAILED("Error while fetching pending messages.", error=str(_e))
    finally:
        if channel:
            channel.close()
        if connection:
            connection.close()

    return []


def _list_pending_messages(channel: BlockingChannel) -> list:
    # declare the queue
    queue_state = channel.queue_declare(queue=QueueName, durable=True, passive=True)
    message_count = queue_state.method.message_count

    messages = []
    for _ in range(message_count):
        method_frame, properties, body = channel.basic_get(
            queue=QueueName, auto_ack=False
        )
        mf: pika.spec.Basic.GetOk = method_frame
        prop: BasicProperties = properties
        msg_task_id = None
        if prop:
            msg_task_id = prop.correlation_id

        if not mf or not msg_task_id:
            continue

        # create a message object
        message = Message()
        message.id = msg_task_id
        message.body = json.loads(body)
        message.properties = prop
        message.method_frame = mf
        messages.append(message)

    return messages


def ack_message(task_id: str) -> bool:
    """
    Acknowledge a message in RabbitMQ with the given task ID.
    This function connects to a RabbitMQ broker, searches for a message with the specified task ID,
    and acknowledges it if found. If the message is acknowledged successfully, it returns True.
    Otherwise, it returns False.

    Args:
        task_id (str): The task ID of the message to be acknowledged.

    Returns:
        bool: True if the message with the specified task ID is acknowledged successfully, False otherwise.

    Raises:
        Exception: If there is an error while connecting to RabbitMQ or acknowledging the message.
    """
    Status.INFO(f"Acknowledging message with Task ID: {task_id}")
    connection = None
    channel = None

    # define the success flag
    success = False

    try:
        # Connect to RabbitMQ
        # Load broker settings
        credentials = pika.PlainCredentials(
            username=GlobalSettings().broker.UserName,
            password=GlobalSettings().broker.Password.get_secret_value(),
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=GlobalSettings().broker.HostName,
                port=GlobalSettings().broker.Port,
                virtual_host=GlobalSettings().broker.VirtualHost,
                credentials=credentials,
            )
        )
        channel = connection.channel()
        # declare the queue
        queue_state = channel.queue_declare(queue=QueueName, durable=True, passive=True)
        message_count = queue_state.method.message_count

        for _ in range(message_count):
            method_frame, properties, _ = channel.basic_get(
                queue=QueueName, auto_ack=False
            )
            mf: pika.spec.Basic.GetOk = method_frame
            prop: BasicProperties = properties
            msg_task_id = prop.correlation_id

            if not mf or not msg_task_id:
                continue

            # filter by task_id
            if msg_task_id == task_id:
                # Acknowledge the message
                channel.basic_ack(mf.delivery_tag)
                Status.SUCCESS(f"Message with Task ID {task_id} acknowledged.")
                success = True
                break

        if not success:
            Status.NOT_FOUND(f"Message with Task ID {task_id} not found.")
    except BaseException as _e:
        Status.FAILED(f"Error while acknowledging Task ID: {task_id}", error=str(_e))
    finally:
        if channel:
            channel.close()
        if connection:
            connection.close()

    return success


def get_broker_url() -> str:
    """
    Retrieves the broker URL from the global settings.

    Returns:
        str: The broker URL.
    """
    uid = GlobalSettings().broker.UserName
    pwd = quote(GlobalSettings().broker.Password.get_secret_value())
    host = GlobalSettings().broker.HostName
    vhost = GlobalSettings().broker.VirtualHost
    port = GlobalSettings().broker.Port
    use_ssl = GlobalSettings().broker.SSL

    if not any([uid, pwd, host, vhost, port]):
        raise ValueError("Broker details are missing in the global settings.")

    protocol = "amqps" if use_ssl else "amqp"
    return f"{protocol}://{uid}:{pwd}@{host}:{port}/{vhost}"
