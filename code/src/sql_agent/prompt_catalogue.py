# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the prompt catalogue in the system."""
from pathlib import Path
from typing import ClassVar
from typing import Optional

from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Sequence
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from src.utils.conf import Setup
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.status import Status

# Define your custom schema
custom_schema = (
    Setup().global_constants.get("DATABASE", {}).get("SCHEMA", "INTELLIGENCE")
)
metadata = MetaData(schema=custom_schema)
Base = declarative_base(metadata=metadata)


class PromptCatalogue(Base):
    """Class to handle the prompt catalogue in the system."""

    __tablename__ = "prompt_catalogue"
    id = Column(
        BigInteger,
        Sequence("prompt_id_sequence", schema=custom_schema, start=1),
        primary_key=True,
        autoincrement=True,
    )
    module = Column(String(255), nullable=False, index=False)
    submodule = Column(String(255), nullable=False, index=False)
    instance_id = Column(String(255), nullable=False, index=False)
    prompt_data = Column(Text, nullable=False, index=False)

    _instance: ClassVar[Instance] = None

    def __init__(self, module: str, submodule: str, instance_id: str):
        self.module = module
        self.submodule = submodule
        self.instance_id = instance_id

        self._instance = GlobalSettings.instance_by_id(self.instance_id)

        # create the table if it does not exist
        if not self.create_table():
            Status.FAILED("Failed to create PromptCatalogue table.")
            # sourcery skip: raise-specific-error
            raise SystemError("Failed to create PromptCatalogue table.")

        # load data from the database
        self.get_prompt_data()

    def __str__(self):
        return f"PromptCatalogue(module={self.module}, submodule={self.submodule}, instance_id={self.instance_id})"

    def upsert(self) -> bool:
        """
        Inserts a new PromptCatalogue record or updates an existing one in the database.

        Returns:
            bool: True if the operation was successful, False otherwise.

        Side Effects:
            - Commits changes to the database session.
            - Rolls back the session in case of exceptions.
            - Logs status messages for success or failure.
        """
        session = self._create_session()
        try:
            if not session:
                return False

            if existing_data := (
                session.query(PromptCatalogue)
                .filter_by(
                    module=self.module,
                    submodule=self.submodule,
                    instance_id=self.instance_id,
                )
                .first()
            ):
                # Update existing data
                existing_data.prompt_data = self.prompt_data
            else:
                # Insert new data
                session.add(self)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            Status.FAILED(
                "Error upserting PromptCatalogue", self, error=e, traceback=False
            )
            return False

    def _create_session(self) -> Optional[Session]:
        if not self._instance:
            Status.NOT_FOUND("Instance not found for PromptCatalogue.")
            return None

        db_uri = self._instance.settings.masterdb.connstring_to_uri()
        engine = create_engine(db_uri)
        return sessionmaker(bind=engine)()

    def get_prompt_data(self) -> Optional[str]:
        """
        Retrieves the prompt data associated with the specified module, submodule, and instance ID.

        Returns:
            Optional[str]: The prompt data string if found; otherwise, None.
        """
        session = self._create_session()
        if not session:
            return None

        data = (
            session.query(PromptCatalogue)
            .filter_by(
                module=self.module,
                submodule=self.submodule,
                instance_id=self.instance_id,
            )
            .first()
        )

        # update the class attributes
        if data:
            self.prompt_data = data.prompt_data

        return data.prompt_data if data else None

    @property
    def filename_path(self) -> Path:
        """Get the path for storing the uploaded filename."""
        file_name = f"prompt_catalog_{self.module}_{self.submodule}_{self.instance_id}_filename.txt"
        return Path(Setup().db_path, self.instance_id, file_name)

    @property
    def exists(self) -> bool:
        """Checks if the prompt catalogue exists for the given module, submodule, and instance ID."""
        prompt_data = self.get_prompt_data()
        return prompt_data is not None

    def save_uploaded_filename(self, filename: str) -> bool:
        """Save the uploaded filename."""
        try:
            with open(self.filename_path, "w", encoding="utf-8") as f:
                f.write(filename)
            return True
        except Exception:
            return False

    def get_uploaded_filename(self) -> Optional[str]:
        """Get the uploaded filename if it exists."""
        try:
            if self.filename_path.exists():
                with open(self.filename_path, encoding="utf-8") as f:
                    return f.read().strip()
        except Exception:  # nosec B110
            pass
        return None

    def get_by_id(self, catalog_id: int) -> Optional["PromptCatalogue"]:
        """
        Retrieve a PromptCatalogue instance by its unique catalog ID.

        This method creates a new database session and queries for a PromptCatalogue
        record matching the provided `catalog_id`. If found, it updates the instance's
        attributes with the retrieved data.

        Args:
            catalog_id (int): The unique identifier of the PromptCatalogue to retrieve.

        Returns:
            Optional[PromptCatalogue]: The current instance with updated attributes if found,
            otherwise None.
        """

        session = self._create_session()
        if not session:
            return None

        if data := (session.query(PromptCatalogue).filter_by(id=catalog_id).first()):
            self.prompt_data = data.prompt_data

        return self

    def create_table(self) -> bool:
        """
        Create the table in the database if it does not exist.
        """
        try:
            engine = create_engine(self._instance.settings.masterdb.connstring_to_uri())
            with engine.connect() as connection:
                if connection.dialect.has_table(
                    connection, self.__tablename__, schema=custom_schema
                ):
                    Status.INFO("PromptCatalogue table already exists.")
                    return True

            Base.metadata.create_all(engine)
            Status.SUCCESS("PromptCatalogue table created successfully.")
            return True
        except Exception as e:
            Status.FAILED(
                "Error creating PromptCatalogue table.", error=e, traceback=True
            )
            return False
