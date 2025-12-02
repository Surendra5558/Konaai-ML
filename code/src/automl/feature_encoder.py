# # Copyright (C) KonaAI - All Rights Reserved
"""This module handles the feature encoding process for the AutoML pipeline."""
from typing import List

from src.automl.custom_transformer import CustomFeaturesTransformer
from src.automl.datetime_encoder import DateTimeEncoder
from src.automl.numeric_encoder import NumericEncoder
from src.automl.onehot_encoder import CustomOneHotEncoder


class CustomFeatureEncoder:
    """
    CustomFeatureEncoder is a class that combines multiple feature encoders
    and transformers to process and encode features for machine learning models.
    Attributes:
    ---------
        onehot_encoder (CustomOneHotEncoder): An instance of CustomOneHotEncoder for encoding categorical features.
        numeric_encoder (NumericEncoder): An instance of NumericEncoder for encoding numeric features.
        custom_features_transformer (CustomFeaturesTransformer): An instance of CustomFeaturesTransformer for custom feature transformations.
    """

    onehot_encoder: CustomOneHotEncoder = None
    numeric_encoder: NumericEncoder = None
    datetime_encoder: DateTimeEncoder = None
    custom_features_transformer: CustomFeaturesTransformer = None

    def get_feature_names_out(self) -> List[str]:
        """
        Get the output feature names after encoding and transformation.
        This method collects feature names from various encoders and transformers
        used in the feature encoding process. It ensures that the feature names
        are unique by converting the list to a set and then back to a list.

        Returns:
            List[str]: A list of unique feature names after encoding and transformation.
        """

        feature_names_ = []
        # get the feature names from the encoders
        if self.onehot_encoder is not None:
            feature_names_.extend(self.onehot_encoder.get_feature_names_out())

        if self.numeric_encoder is not None:
            feature_names_.extend(self.numeric_encoder.get_feature_names_out())

        if self.custom_features_transformer is not None:
            feature_names_.extend(
                self.custom_features_transformer.get_feature_names_out()
            )

        if self.datetime_encoder is not None:
            feature_names_.extend(self.datetime_encoder.get_feature_names_out())

        return list(set(feature_names_))

    def get_column_names_in(self) -> List[str]:
        """
        Retrieve the column names from the encoders.
        This method collects the column names from the onehot_encoder, numeric_encoder,
        and custom_features_transformer if they are not None. It combines these column
        names into a single list, removes duplicates, and returns the list.

        Returns:
            List[str]: A list of unique column names from the encoders.
        """
        column_names_ = []
        # get the feature names from the encoders
        if self.onehot_encoder is not None:
            column_names_.extend(self.onehot_encoder.columns)

        if self.numeric_encoder is not None:
            column_names_.extend(self.numeric_encoder.columns)

        if self.custom_features_transformer is not None:
            column_names_.extend(self.custom_features_transformer.columns)

        if self.datetime_encoder is not None:
            column_names_.extend(self.datetime_encoder.columns)

        return list(set(column_names_))
