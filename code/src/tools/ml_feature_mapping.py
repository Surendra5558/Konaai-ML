# # Copyright (C) KonaAI - All Rights Reserved
"""This module maps ML feature names with application algorithm test names"""
import json
import os
from typing import Dict
from typing import Union

from src.tools.utils import config
from src.utils.conf import Setup
from src.utils.status import Status


class MLFeatureMappings:
    """
    MLFeatureMappings is a utility class for managing and retrieving machine learning feature mappings from a JSON configuration file.
    This class provides methods to:
    - Load feature mappings from a JSON file specified in the configuration.
    - Retrieve the PatternID for a given feature name.
    - Retrieve the machine learning feature name and its long description for a given PatternID.
    - Retrieve the long description for a given feature name.
    Attributes:
    ----------
        all_mappings (dict): A dictionary containing all feature mappings loaded from the JSON file.
    """

    def __init__(self):
        self.all_mappings = self.get_feature_mappings()

    def get_feature_mappings(self):
        """reading json file with columns name

        Returns:
            _type_: json file
        """
        try:
            feature_mapping_file_path = os.path.join(
                Setup().assets_path, config.get("feature_mappings", "file_name")
            )

            with open(feature_mapping_file_path, encoding="utf-8") as _f:
                return json.load(_f)
        except BaseException as _e:
            Status.FAILED("Error in reading feature mappings file", error=str(_e))
            return {}

    def get_pattern_id(self, feature_name: str) -> Union[str, None]:
        """
        Retrieve the PatternID for a given feature name.
        Args:
        -----
            feature_name (str): The name of the feature for which to retrieve the PatternID.

        Returns:
            str or None: The PatternID if found, otherwise None.

        Logs:
            Logs an error message if no PatternID is found for the given feature name.
        """
        return self._extract_from_json(
            feature_name, "PatternID", "No PatternID found for feature name: "
        )

    def get_ml_feature_name(self, pattern_id: str) -> Dict[str, str]:
        """
        Retrieve the machine learning feature name and description for a given pattern ID.
        Args:
        ----
            pattern_id (str): The pattern ID to search for in the feature mappings.

        Returns:
            Dict[str, str]: A dictionary containing the pattern ID and its corresponding
                            long description if found, otherwise an empty dictionary.

        Logs:
            Logs an error message if the pattern ID is not found in the feature mappings.
        """

        # check if the application test name is present in the feature mappings
        if pattern_id not in [v.get("PatternID") for v in self.all_mappings.values()]:
            Status.FAILED("No feature name found for PatternID", pattern_id=pattern_id)
            return {}

        return (
            {
                "PatternID": next(iter(filtered_mappings.keys())),
                "PatternDescriptionLong": next(iter(filtered_mappings.values())),
            }
            if (
                filtered_mappings := {
                    k: v.get("PatternDescriptionLong")
                    for k, v in self.all_mappings.items()
                    if v.get("PatternID") == pattern_id
                }
            )
            else {}
        )

    def get_description(self, feature_name: str) -> Union[str, None]:
        """
        Retrieve the description for a given feature name.
        Args:
        ----
            feature_name (str): The name of the feature for which the description is to be retrieved.

        Returns:
            Union[str, None]: The description of the feature if found, otherwise None.

        Logs:
            Logs an error message if no description is found for the given feature name.
        """

        return self._extract_from_json(
            feature_name,
            "PatternDescriptionLong",
            "No Description found for feature name: ",
        )

    # TODO Rename this here and in `get_pattern_id` and `get_description`
    def _extract_from_json(
        self, feature_name: str, arg1: str, arg2: str
    ) -> Union[str, None]:
        """
        Extracts a value from a JSON-like mapping based on the provided feature name and argument.

        Args:
        -----
            feature_name (str): The name of the feature to extract.
            arg1 (str): The first argument used to locate the value in the mapping.
            arg2 (str): The second argument used for logging purposes if the feature is not found.

        Returns:
            The extracted value if found, otherwise None.
        """
        if result := self.all_mappings.get(feature_name, {}).get(arg1):
            return result
        Status.FAILED(f"{arg2}{feature_name}")
        return None
