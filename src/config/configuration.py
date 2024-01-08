from src.constants import *
from src.utils.common import read_yaml
from src import logger
from src.entity.config_entity import DataTokenizationConfig, DataEncodingConfig, HeadModelTrainingConfig, HeadModelFoldTrainingConfig, PredictingConfig, StackingModelTrainingConfig



class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for reading and providing 
    configuration settings needed for various stages of the data pipeline.

    Attributes:
    - config (dict): Dictionary holding configuration settings from the config file.
    - params (dict): Dictionary holding parameter values from the params file.
    - schema (dict): Dictionary holding schema information from the schema file.
    """
    
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH, 
                 schema_filepath=SCHEMA_FILE_PATH):
        """
        Initializes the ConfigurationManager with configurations, parameters, and schema.

        Parameters:
        - config_filepath (str): Filepath to the configuration file.
        - params_filepath (str): Filepath to the parameters file.
        - schema_filepath (str): Filepath to the schema file.
        """
        self.config = self._read_config_file(config_filepath, "config")
        self.params = self._read_config_file(params_filepath, "params")
        self.schema = self._read_config_file(schema_filepath, "schema")

    
    def _read_config_file(self, filepath: str, config_name: str) -> dict:
        """
        Reads and returns the content of a configuration file.

        Parameters:
        - filepath (str): The file path to the configuration file.
        - config_name (str): Name of the configuration (used for logging purposes).

        Returns:
        - dict: Dictionary containing the configuration settings.

        Raises:
        - Exception: An error occurred reading the configuration file.
        """
        try:
            return read_yaml(filepath)
        except Exception as e:
            logger.error(f"Error reading {config_name} file: {filepath}. Error: {e}")
            raise

    
    def get_data_tokenization_config(self) -> DataTokenizationConfig:
        """
        Extracts and returns data tokenization configuration settings as a DataTokenizationConfig object.

        Returns:
        - DataTokenizationConfig: Object containing data tokenization configuration settings.

        Raises:
        - AttributeError: The 'data_tokenization' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_tokenization
            return DataTokenizationConfig(
                max_length=config.max_length,
                model_checkpoint=config.model_checkpoint
            )
        except AttributeError as e:
            logger.error("The 'data_tokenization' attribute does not exist in the config file.")
            raise e


    def get_data_encoding_config(self) -> DataEncodingConfig:
        """
        Extracts and returns data encoding configuration settings as a DataEncodingConfig object.

        Returns:
        - DataEncodingConfig: Object containing data encoding configuration settings.

        Raises:
        - AttributeError: The 'data_encoding' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_encoding
            return DataEncodingConfig(
                n_labels=config.n_labels,
                batch_size=config.batch_size,
                n_batches_per_save=config.n_batches_per_save,
                model_checkpoint=config.model_checkpoint,
                start_batch=config.start_batch,
                end_batch=config.end_batch
            )
        except AttributeError as e:
            logger.error("The 'data_encoding' attribute does not exist in the config file.")
            raise e


    def get_head_model_training_config(self) -> HeadModelTrainingConfig:
        """
        Extracts and returns head model training configuration settings as a HeadModelTrainingConfig object.

        Returns:
        - HeadModelTrainingConfig: Object containing head model training configuration settings.

        Raises:
        - AttributeError: The 'head_model_training' attribute does not exist in the config file.
        """
        try:
            config = self.config.head_model_training
            return HeadModelTrainingConfig(
                embedding_file_size=config.embedding_file_size,
                validation_embedding_id=config.validation_embedding_id,
                min_train_embedding_id=config.min_train_embedding_id,
                max_train_embedding_id=config.max_train_embedding_id,
                training_configs=config.training_configs
            )
        except AttributeError as e:
            logger.error("The 'head_model_training' attribute does not exist in the config file.")
            raise e


    def get_head_model_fold_training_config(self) -> HeadModelFoldTrainingConfig:
        """
        Extracts and returns head model fold-wise training configuration settings as a HeadModelFoldTrainingConfig object.

        Returns:
        - HeadModelFoldTrainingConfig: Object containing head model fold-wise training configuration settings.

        Raises:
        - AttributeError: The 'head_model_fold_training' attribute does not exist in the config file.
        """
        try:
            config = self.config.head_model_fold_training
            return HeadModelFoldTrainingConfig(
                embedding_file_size=config.embedding_file_size,
                validation_embedding_id=config.validation_embedding_id,
                min_train_embedding_id=config.min_train_embedding_id,
                max_train_embedding_id=config.max_train_embedding_id,
                training_configs=config.training_configs,
                folds_ids_range=config.folds_ids_range
            )
        except AttributeError as e:
            logger.error("The 'head_model_fold_training' attribute does not exist in the config file.")
            raise e


    def get_stacking_model_training_config(self) -> StackingModelTrainingConfig:
        """
        Extracts and returns stacking model training configuration settings as a StackingModelTrainingConfig object.

        Returns:
        - StackingModelTrainingConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'stacking_model_training' attribute does not exist in the config file.
        """
        try:
            config = self.config.stacking_model_training
            return StackingModelTrainingConfig(
                embedding_file_size=config.embedding_file_size,
                min_train_embedding_id=config.min_train_embedding_id,
                max_train_embedding_id=config.max_train_embedding_id,
                head_models=config.head_models,
                models=config.models,
                scoring_metric=config.scoring_metric,
                swap_labels=config.swap_labels,
                n_folds=config.n_folds,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
                optimization_direction=config.optimization_direction,
                n_trials=config.n_trials
            )
        except AttributeError as e:
            logger.error("The 'stacking_model_training' attribute does not exist in the config file.")
            raise e
            

    def get_predicting_config(self) -> PredictingConfig:
        """
        Extracts and returns predicting configuration settings as a PredictingConfig object.

        Returns:
        - PredictingConfig: Object containing predicting configuration settings.

        Raises:
        - AttributeError: The 'predicting' attribute does not exist in the config file.
        """
        try:
            config = self.config.predicting
            return PredictingConfig(
                embedding_file_size=config.embedding_file_size,
                embedding_ids=config.embedding_ids,
                head_models=config.head_models,
                second_level_model=config.second_level_model,
                metrics_names=config.metrics_names
            )
        except AttributeError as e:
            logger.error("The 'predicting' attribute does not exist in the config file.")
            raise e