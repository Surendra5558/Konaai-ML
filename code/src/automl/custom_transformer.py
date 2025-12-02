# # Copyright (C) KonaAI - All Rights Reserved
"""Custom transformer to add features to the data"""
import importlib
import inspect
import pkgutil
import sys
from typing import List
from typing import Union

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.tools.dask_tools import compute
from src.tools.dask_tools import validate_column
from src.utils.base_feature import BaseFeatureBuilder
from src.utils.file_mgmt import file_handler
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm


class CustomFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering and anomaly detection.
    Attributes:
    -----------
        anomaly_prefix (str): Prefix for anomaly features.
        valid_feature_builders (list): List of valid feature builders.
        feature_names_ (list): List of feature names.
        columns (list): List of columns.
        index (str): Index column name.
    """

    anomaly_prefix = "Anmly_"

    def __init__(self, submodule: Submodule, index_column: str = None):
        self.valid_feature_builders = []
        self.feature_names_ = []
        self.columns = None
        self.index = index_column
        self.submodule = submodule

    def features_names_out(self) -> List[str]:
        """
        Returns the list of feature names after transformation.
        This method returns a list of unique feature names that are generated
        after the transformation process. The feature names are stored in the
        instance variable `_features_names_out`.

        Returns:
            List[str]: A list of unique feature names.
        """

        return list(set(self._features_names_out))

    def get_feature_builders_from_module(self, module_name) -> List[BaseFeatureBuilder]:
        """
        Retrieves all classes from the specified module that are subclasses of BaseFeatureBuilder,
        excluding the BaseFeatureBuilder class itself.
        Args:
        ----
            module_name (str): The name of the module to import and inspect.
        Returns:
            List[BaseFeatureBuilder]: A list of classes that are subclasses of BaseFeatureBuilder.
        """
        # Import the module
        module = importlib.import_module(module_name)

        # Get all the classes in the module
        classes = inspect.getmembers(module, inspect.isclass)

        return [
            class_info[1]
            for class_info in classes
            if issubclass(class_info[1], BaseFeatureBuilder)
            and class_info[0] != BaseFeatureBuilder.__name__
        ]

    def get_feature_builders_from_package(self, package_name) -> List:
        """
        Retrieves feature builder classes from all modules within a specified package.

        Args:
        -----
            package_name (str): The name of the package to search for feature builders.

        Returns:
            List: A list of feature builder classes found within the specified package.
        """
        # Import the package
        package = importlib.import_module(package_name)
        modules = [
            module_info.name for module_info in pkgutil.iter_modules(package.__path__)
        ]
        feature_builders = []
        for module_name in modules:
            feature_builders += self.get_feature_builders_from_module(
                f"{package_name}.{module_name}"
            )

        return feature_builders

    def get_feature_builders(self) -> List[BaseFeatureBuilder]:
        """
        Retrieves a list of feature builders from specified packages.

        This method collects feature builders from the 'src.p2p.features' and 'src.te.features' packages.

        Returns:
            List[BaseFeatureBuilder]: A list of feature builder instances.
        """
        # List all the features in the features folder
        builders = []
        feature_packages = ["src.p2p.features", "src.te.features"]

        for package_name in feature_packages:
            try:
                builders += self.get_feature_builders_from_package(package_name)
            except ModuleNotFoundError:
                Status.WARNING(f"Feature package '{package_name}' not found. Skipping.")
            except Exception as e:
                Status.FAILED(f"Error loading features from {package_name}: {e}")

        return builders

    def get_anomaly_builders(self, builders: List) -> List[BaseFeatureBuilder]:
        """
        Filters and returns a list of anomaly builders from the provided list of builders.

        Args:
        -----
            builders (List): A list of builder objects to filter.

        Returns:
            List[BaseFeatureBuilder]: A list of builders whose module name contains "anomaly_".
        """
        return [builder for builder in builders if "anomaly_" in builder.__module__]

    def validate_index(
        self, _df: dd.DataFrame, index: str, move_index_as_column=False
    ) -> Union[dd.DataFrame, None]:
        """
        Validates and sets the index of a Dask DataFrame.
        This method checks if the specified index column is present in the DataFrame,
        and if necessary, moves the index to a column or sets the specified column as the index.
        Parameters:
        -----------
        _df : dd.DataFrame
            The Dask DataFrame to validate and modify.
        index : str
            The name of the column to be used as the index.
        move_index_as_column : bool, optional (default=False)
            If True, moves the current index to a column if it matches the specified index.
        Returns:
        --------
        dd.DataFrame or None
            The modified DataFrame with the correct index set, or None if the index column is not present.
        """
        index = validate_column(index, _df)
        if not index:
            # index column is must for all features
            Status.NOT_FOUND("Index column not present in the data")
            return None

        # check if the index column already an index
        if (
            _df.index.name is not None
            and _df.index.name.lower() == index.lower()
            and move_index_as_column
        ):
            Status.INFO(f"Moving {index} as a column")

            # move index as a column
            _df = _df.reset_index(drop=False)
        # check if the index column is not present in the data
        else:
            _df = _df.set_index(index, drop=True)

        # return _df or None
        return _df

    def execute_builder(
        self, builder: BaseFeatureBuilder, X: dd.DataFrame, index: str
    ) -> Union[dd.DataFrame, None]:
        """
        Executes the feature builder on the provided Dask DataFrame.

        This method performs the following steps:
        1. Saves the input DataFrame to disk in Parquet format.
        2. Initializes the feature builder with the saved data path and index column.
        3. Loads the data using the builder.
        4. Validates the index of the loaded data.
        5. Runs the feature builder to generate features.
        6. Validates the index of the generated features.
        7. Writes the output features to disk and loads them back into a Dask DataFrame.
        Args:
        -----
            builder (BaseFeatureBuilder): The feature builder to execute.
            X (dd.DataFrame): The input Dask DataFrame.
            index (str): The name of the index column.
        Returns:
            Union[dd.DataFrame, None]: The resulting Dask DataFrame with generated features, or None if an error occurs or no data is generated.
        """
        try:
            # save the data to disk
            _, data_path = file_handler.get_new_file_name("parquet")

            X.to_parquet(
                data_path,
                engine="pyarrow",
                compression="snappy",
                write_index=True,
                compute=True,
            )

            # create builder object
            builder = builder(input_data_path=data_path, index_column=index)

            Status.INFO(
                f"Loading feature {builder.__module__}.{builder.__class__.__name__}"
            )

            # load data
            df = builder.load_data(self.submodule)

            if df is None or len(df) == 0:
                Status.INFO(
                    f"Feature {builder.__module__}.{builder.__class__.__name__} has no data",
                    self.submodule,
                )
                return None

            # validate index
            df = self.validate_index(df, index, move_index_as_column=True)

            # run the builder
            Status.INFO(
                f"Running feature {builder.__module__}.{builder.__class__.__name__}"
            )
            df = builder.run(df)

            # validate data
            if df is None or len(df) == 0:
                Status.INFO(
                    f"Feature {builder.__module__}.{builder.__class__.__name__} has no data",
                    self.submodule,
                )
                return None

            # validate index
            df = self.validate_index(df, index, move_index_as_column=False)
            output_path = builder.write_output(df, self.submodule)

            # load the data
            return dd.read_parquet(output_path)
        except Exception as e:
            Status.FAILED(
                f"Error in {builder.__module__}.{builder.__class__.__name__}",
                self.submodule,
                error=e,
            )
            return None

    def transform_with_builder(
        self,
        X: dd.DataFrame,
        builder: BaseFeatureBuilder,
        index: str,
        feature_name_prefix: str = None,
    ) -> Union[dd.DataFrame, None]:
        """
        Transforms the input DataFrame using a specified feature builder.
        Parameters:
        -----------
        X : dd.DataFrame
            The input Dask DataFrame to be transformed.
        builder : BaseFeatureBuilder
            The feature builder class used to transform the DataFrame.
        index : str
            The name of the index column in the DataFrame.
        feature_name_prefix : str, optional
            A prefix to add to the names of the generated features (default is None).
        Returns:
        --------
        dd.DataFrame or None
            The transformed DataFrame with new features, or None if the required columns are not present in the input DataFrame or if the builder execution fails.
        """
        # create builder object
        builder_obj = builder(input_data_path=None, index_column=index)

        # check if the data has all the columns needed by the builder
        required_cols = builder_obj.input_cols
        if any(col not in X.columns for col in required_cols):
            return None

        # execute the builder
        result_df = self.execute_builder(builder, X, index=index)
        if result_df is None:
            return None

        # prepare rename mapper
        if feature_name_prefix:
            column_mapper = {
                col: f"{feature_name_prefix}{col}" for col in result_df.columns
            }
            result_df = result_df.rename(columns=column_mapper)
            # update feature names
            self.feature_names_ += column_mapper.values()

        return compute(result_df)

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names for transformation.

        Returns:
            List[str]: A list of feature names after transformation.
        """
        return self.feature_names_

    def fit(self, X: dd.DataFrame, y=None):  # pylint: disable=unused-argument
        """
        Fits the custom features transformer to the data.
        This method validates and selects applicable anomaly feature builders based on the input data.
        It iterates through all anomaly builders, checks if they have the required input columns,
        and adds the valid builders to the list of valid feature builders.
        Parameters:
        -----------
        X (pd.DataFrame): The input data to fit the transformer.
        y (pd.Series, optional): The target values (default is None).

        Returns:
        self: Returns the instance of the transformer.
        """
        Status.INFO("Fitting custom features transformer")

        all_builders = self.get_feature_builders()
        anomaly_builders = self.get_anomaly_builders(all_builders)

        # find valid anomaly builders
        self.columns = []
        for builder in tqdm(
            anomaly_builders,
            desc="Validating applicable anomaly features",
            file=sys.stdout,
        ):
            # create builder object
            builder_obj: BaseFeatureBuilder = builder(
                input_data_path=None, index_column=self.index
            )

            # check if builder has input_cols attribute
            if not hasattr(builder_obj, "input_cols"):
                continue

            # check if the data has all the columns needed by the builder
            if all(col in X.columns for col in builder_obj.input_cols):
                # add the builder to the list of valid builders
                self.valid_feature_builders.append(builder)
                # add the columns to the list of input columns
                self.columns.extend(builder_obj.input_cols)

        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Transforms the input DataFrame by processing anomaly features using valid anomaly builders.

        Parameters:
        X (pd.DataFrame): The input DataFrame containing the features to be transformed.

        Returns:
        pd.DataFrame: The transformed DataFrame with new anomaly features added.

        The method performs the following steps:
        1. Logs the start of the transformation process.
        2. Identifies the original columns in the input DataFrame.
        3. Retrieves all feature builders and filters them to get valid anomaly builders.
        4. Iterates over the valid anomaly builders and transforms the input DataFrame using each builder.
        5. Merges the transformation results with the original DataFrame.
        6. Identifies and logs the new columns added to the DataFrame.
        """
        Status.INFO("Transforming custom features transformer")
        # find original columns
        original_columns = X.columns

        self.index = validate_column(self.index, X)

        # process the features
        for builder in tqdm(
            self.valid_feature_builders,
            desc="Processing anomaly feature builders",
            file=sys.stdout,
        ):
            result_df = self.transform_with_builder(
                X, builder, self.index, feature_name_prefix=self.anomaly_prefix
            )
            if result_df is not None and len(result_df) > 0:
                # merge the result with the original data
                X = compute(X.merge(result_df, how="left", on=self.index))

        # find updated columns
        updated_columns = X.columns

        # find the new columns
        new_columns = list(set(updated_columns) - set(original_columns))
        Status.INFO(f"Total {len(new_columns)} new features added")

        # set index
        X = X.set_index(self.index)

        return X
