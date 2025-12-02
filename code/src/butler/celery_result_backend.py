# # Copyright (C) KonaAI - All Rights Reserved
"""This module creates the Celery backend tables and Model Monitoring table in the database."""
from datetime import datetime
from typing import ClassVar
from typing import Optional

from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import LargeBinary
from sqlalchemy import MetaData
from sqlalchemy import Sequence
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema
from src.utils.conf import Setup
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.status import Status

# Define your custom schema
custom_schema = (
    Setup().global_constants.get("DATABASE", {}).get("SCHEMA", "INTELLIGENCE")
)

# Create a MetaData instance with the custom schema
metadata = MetaData(schema=custom_schema)
Base = declarative_base(metadata=metadata)


class TaskResultTable(Base):
    """
    Represents a task result in the database.

    Attributes:
    ---------
        id (int): The unique identifier of the task result.
        task_id (str): The unique identifier of the task.
        status (str): The status of the task result.
        result (str): The result of the task.
        date_done (datetime): The date and time when the task was completed.
        traceback (str): The traceback information if an error occurred during the task execution.
        children (str): The children tasks associated with this task result.
    """

    __tablename__ = "celery_taskmeta"
    id = Column(
        BigInteger,
        Sequence("task_id_sequence", schema=custom_schema, start=1),
        primary_key=True,
        autoincrement=True,
    )
    task_id = Column(String(255), unique=True, index=True)
    status = Column(String(50))
    result = Column(LargeBinary, nullable=True)
    date_done = Column(DateTime)
    traceback = Column(Text, nullable=True)
    children = Column(Text, nullable=True)
    date_submit = Column(DateTime, nullable=True)

    @classmethod
    def upsert_task(cls, task_id: str, status: str) -> bool:
        """
        Upserts a task in the database with the given task ID and status.
        If a task with the given task ID already exists, its status is updated.
        Otherwise, a new task is created with the provided task ID and status.
        Args:
        ----
            task_id (str): The unique identifier of the task.
            status (str): The status of the task.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            url = get_backend_url(for_celery=False)
            engine = create_engine(url)
            session = sessionmaker(bind=engine)()

            if task := session.query(cls).filter_by(task_id=task_id).first():
                task.status = status
            else:
                task = cls(task_id=task_id, status=status)
                session.add(task)
            session.commit()
            return True
        except BaseException:
            return False

    @classmethod
    def get_task_status(cls, task_id: str) -> str:
        """
        Retrieve the status of a task given its task ID.

        Args:
            task_id (str): The ID of the task whose status is to be retrieved.

        Returns:
            str: The status of the task. Returns "FAILURE" if the task is not found or if an exception occurs.
        """
        try:
            url = get_backend_url(for_celery=False)
            engine = create_engine(url)
            session = sessionmaker(bind=engine)()
            task = session.query(cls).filter_by(task_id=task_id).first()
            return str(task.status).upper() if task else "FAILURE"
        except BaseException:
            return "FAILURE"

    @classmethod
    def task_exists(cls, task_id: str) -> bool:
        """
        Check if a task exists in the database.

        Args:
            task_id (str): The unique identifier of the task.

        Returns:
            bool: True if the task exists, False otherwise.
        """
        try:
            url = get_backend_url(for_celery=False)
            engine = create_engine(url)
            session = sessionmaker(bind=engine)()
            return session.query(cls).filter_by(task_id=task_id).first() is not None
        except BaseException:
            return False


class TaskSetResult(Base):
    """
    Represents the result of a task set in the application.

    Attributes:
    ----------
        id (int): The primary key of the task set result.
        taskset_id (str): The unique identifier of the task set.
        result (str): The result of the task set.
        date_done (datetime): The date and time when the task set was completed.
    """

    __tablename__ = "celery_tasksetmeta"
    id = Column(BigInteger, primary_key=True)
    taskset_id = Column(String(255), unique=True, index=True)
    result = Column(LargeBinary, nullable=True)
    date_done = Column(DateTime)


class ModelMonitoring(Base):
    """Class to handle model monitoring in the system."""

    __tablename__ = "model_monitoring"
    id = Column(
        BigInteger,
        Sequence("model_monitoring_id_sequence", schema=custom_schema, start=1),
        primary_key=True,
        autoincrement=True,
    )
    task_id = Column(String(255), unique=True, index=True)
    instance_id = Column(String(255), nullable=False, index=True)
    module = Column(String(255), nullable=False, index=False)
    submodule = Column(String(255), nullable=False, index=False)
    model_name = Column(String(255), nullable=False, index=False)
    active_f1_score = Column(Float, nullable=True)
    new_f1_score = Column(Float, nullable=True)
    f1_score_change = Column(Float, nullable=True)
    concern_count = Column(BigInteger, nullable=True)
    no_concern_count = Column(BigInteger, nullable=True)
    date_run = Column(DateTime, nullable=True)

    _instance: ClassVar[Instance] = None

    def __init__(
        self,
        task_id: str,
        instance_id: str,
        module: str,
        submodule: str,
        model_name: str,
        active_f1_score: float = None,
        new_f1_score: float = None,
        f1_score_change: float = None,
        concern_count: int = None,
        no_concern_count: int = None,
        date_run: datetime = None,
    ):
        self.task_id = task_id
        self.instance_id = instance_id
        self.module = module
        self.submodule = submodule
        self.model_name = model_name
        self.active_f1_score = active_f1_score
        self.new_f1_score = new_f1_score
        self.f1_score_change = f1_score_change
        self.concern_count = concern_count
        self.no_concern_count = no_concern_count
        self.date_run = date_run

        self._instance = GlobalSettings.instance_by_id(self.instance_id)

        # create the table if it does not exist
        if not create_tables():
            Status.FAILED("Failed to create ModelMonitoring table.")
            # sourcery skip: raise-specific-error
            raise SystemError("Failed to create ModelMonitoring table.")

    def __str__(self):
        return f"ModelMonitoring(task_id={self.task_id}, module={self.module}, submodule={self.submodule}, model_name={self.model_name})"

    def upsert(self) -> bool:
        """
        Inserts a new ModelMonitoring record or updates an existing one in the database.
        It uses the attributes already set on the instance (self).

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        session = self._create_session()
        try:
            if not session:
                return False

            if existing_record := (
                session.query(ModelMonitoring).filter_by(task_id=self.task_id).first()
            ):
                # Update existing record
                existing_record.task_id = self.task_id
                existing_record.instance_id = self.instance_id
                existing_record.module = self.module
                existing_record.submodule = self.submodule
                existing_record.model_name = self.model_name
                existing_record.active_f1_score = self.active_f1_score
                existing_record.new_f1_score = self.new_f1_score
                existing_record.f1_score_change = self.f1_score_change
                existing_record.concern_count = self.concern_count
                existing_record.no_concern_count = self.no_concern_count
                existing_record.date_run = self.date_run
            else:
                # Insert new record (This will correctly insert the new instance 'self')
                session.add(self)

            session.commit()
            Status.SUCCESS("ModelMonitoring record upserted successfully.", self)
            return True
        except Exception as e:
            session.rollback()
            Status.FAILED(
                "Error upserting ModelMonitoring record", self, error=e, traceback=False
            )
            return False

    def _create_session(self) -> Optional[Session]:
        if not self._instance:
            Status.NOT_FOUND("Instance not found for ModelMonitoring.")
            return None

        db_uri = self._instance.settings.masterdb.connstring_to_uri()
        engine = create_engine(db_uri)
        return sessionmaker(bind=engine)()

    def get_record(self) -> Optional["ModelMonitoring"]:
        """
        Retrieves the model monitoring record for the current task_id.

        Returns:
            Optional[ModelMonitoring]: The ModelMonitoring instance with data if found; otherwise, None.
        """
        session = self._create_session()
        if not session:
            return None

        data = session.query(ModelMonitoring).filter_by(task_id=self.task_id).first()

        # update the class attributes if data found
        if data:
            self.task_id = data.task_id
            self.instance_id = data.instance_id
            self.module = data.module
            self.submodule = data.submodule
            self.model_name = data.model_name
            self.active_f1_score = data.active_f1_score
            self.new_f1_score = data.new_f1_score
            self.f1_score_change = data.f1_score_change
            self.concern_count = data.concern_count
            self.no_concern_count = data.no_concern_count
            self.date_run = data.date_run

        return data


def get_backend_url(for_celery=True):
    """
    Retrieves the backend URL from the database handler.

    Returns:
        str: The backend URL.
    """
    if conn := GlobalSettings().workerdb.connstring_to_uri():
        return conn.replace("mssql", "db+mssql") if for_celery else conn
    Status.FAILED("Can not get the backend URL. Check global configuration.")
    return ""


def get_all_model_monitoring_records():
    """
    Retrieve all model monitoring records.

    Returns:
        list[ModelMonitoring]: List of ModelMonitoring records
    """
    try:
        url = get_backend_url(for_celery=False)
        engine = create_engine(url)
        session = sessionmaker(bind=engine)()
        records = session.query(ModelMonitoring).all()
        return records
    except Exception as e:
        Status.FAILED("Error retrieving all records", error=e)
        return []


def create_tables():
    """
    Creates all tables (Celery backend and Model Monitoring) in the database.

    Returns:
        bool: True if the tables are created successfully, False otherwise.
    """
    try:
        url = get_backend_url(for_celery=False)
        engine = create_engine(url)
        connection = engine.connect()

        # check if able to connect to the database
        if not connection:
            Status.FAILED(
                "Can not connect to the database. Check global configuration."
            )
            return False

        # check if schema does not exist
        if not engine.dialect.has_schema(connection, custom_schema):
            Status.NOT_FOUND(
                f"Schema does not exist. Creating schema {custom_schema} in worker database."
            )
            connection.execute(CreateSchema(custom_schema))
            connection.commit()
            Status.SUCCESS(f"Schema {custom_schema} created in worker database.")
        else:
            Status.INFO(f"Schema {custom_schema} already exists in worker database.")

        # create all tables
        Base.metadata.create_all(engine)
        Status.SUCCESS(
            "All tables (Celery backend and Model Monitoring) created successfully."
        )

        return True
    except BaseException as _e:
        Status.FAILED(
            "Can not create tables. Check global configuration.",
            error=_e,
        )
        return False


if __name__ == "__main__":
    create_tables()
