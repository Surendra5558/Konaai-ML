# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to create the ml parameters class"""
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class MLParameters(BaseModel):
    """
    MLParameters is a Pydantic model that encapsulates configuration parameters for machine learning workflows.
        missing_threshold (int): The percentage threshold (0-100) for acceptable missing values in features.
        correlation_threshold (int): The percentage threshold (0-100) for correlation filtering; features with correlation above this value are dropped.
        test_size (int): The percentage (0-100) of data reserved for testing; the remainder is used for training.
        n_splits (int): The number of splits (0-10) for cross-validation.
        need_synthetic (bool): Indicates whether synthetic data should be generated for training.

    Config:
        extra (str): Allows extra fields not defined in the model ("allow").
        frozen (bool): Indicates if the configuration is immutable (False).

    Methods:
        get_field_name_from_title(title: str) -> str:
            Retrieves the actual field name corresponding to a given title defined in the Field metadata.
            Returns the field name if found, otherwise None.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
    )

    missing_threshold: int = Field(
        80,
        title="Missing Threshold",
        description="The % threshold for missing values that are acceptable",
        ge=0,
        le=100,
    )
    correlation_threshold: int = Field(
        90,
        title="Correlation Threshold",
        description="The % threshold for correlation filtering. Features with correlation greater than this value will be dropped.",
        ge=0,
        le=100,
    )
    test_size: int = Field(
        20,
        title="Test Size",
        description="The % of the test data to validate results. The remaining data will be used for training",
        ge=0,
        le=100,
    )
    n_splits: int = Field(
        5,
        title="Number of Training Iterations",
        description="The number of splits and iterations to use for cross validation",
        ge=0,
        le=10,
    )
    need_synthetic: bool = Field(
        True,
        title="Need Synthetic Data",
        description="Whether to generate synthetic data or not for training",
    )

    @classmethod
    def get_field_name_from_title(cls, title: str) -> str:
        """
        Retrieves the actual field name from the title defined in the Field.

        Args:
            title: The title of the field.

        Returns:
            The corresponding field name, or None if not found.
        """
        for field_name, field in cls.model_fields.items():
            if field.title == title:
                return field_name
        return None
