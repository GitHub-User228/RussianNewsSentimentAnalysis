from dataclasses import dataclass



@dataclass(frozen=True)
class DataTokenizationConfig:
    """
    Configuration for the data tokenization process.
    
    Attributes:
    - max_length: Maximum length of a text to be considered (if more, than such text will be shrinked).
    - model_checkpoint: Model to be used to encode data (from hugging face)
    """
    
    max_length: int
    model_checkpoint: str



@dataclass(frozen=True)
class DataEncodingConfig:
    """
    Configuration for the data encoding process.
    
    Attributes:
    - n_labels: Number of classes (labels) in data to be encoded
    - batch_size: Batch size
    - n_batches_per_save: Number of batches to be saved in a single file
    - model_checkpoint: Model to be used to encode data (from hugging face)
    - start_batch: Id of batch from which encoding must be started
    - end_batch: Id of batch up to which encoding must be done
    """

    n_labels: int
    batch_size: int
    n_batches_per_save: int
    model_checkpoint: str
    start_batch: int
    end_batch: int



@dataclass(frozen=True)
class HeadModelTrainingConfig:
    """
    Configuration for the head model training process.
    
    Attributes:
    - embedding_file_size: Number of observations in a single file with embeddings
    - validation_embedding_id: Id of file to be used as validation data.
    - min_train_embedding_id: Min id of file to be used as training data.
    - max_train_embedding_id: Max id of file to be used as training data.
    - training_configs: Dictionary with default training configs
    """
    
    embedding_file_size: int
    validation_embedding_id: int
    min_train_embedding_id: int
    max_train_embedding_id: int
    training_configs: dict



@dataclass(frozen=True)
class HeadModelFoldTrainingConfig:
    """
    Configuration for the head model fold-wise training process.
    
    Attributes:
    - embedding_file_size: Number of observations in a single file with embeddings
    - validation_embedding_id: Id of file to be used as validation data.
    - min_train_embedding_id: Min id of file to be used as training data.
    - max_train_embedding_id: Max id of file to be used as training data.
    - training_configs: Dictionary with default training configs
    - folds_ids_range: List of lists with range of embedding ids for folds.
    """
    
    embedding_file_size: int
    validation_embedding_id: int
    min_train_embedding_id: int
    max_train_embedding_id: int
    training_configs: dict
    folds_ids_range: list



@dataclass(frozen=True)
class StackingModelTrainingConfig:
    """
    Configuration for the stacking model training process.
    
    Attributes:
    - embedding_file_size: Number of observations in a single file with embeddings
    - min_train_embedding_id: Min id of file to be used as training data.
    - max_train_embedding_id: Max id of file to be used as training data.
    - head_models: List of head models, from which logits will be used as training data for a second level model
    - models: List of second level models to be trained and tuned
    - scoring_metric: Name of the metric to be used for evaluations
    - swap_labels: Whether to swap 0 and 1 class labels for tuning process 
                   (in case, when scoring_metric needs to be calculated for 0 class)
    - n_folds: Number of folds in cross validation process (when tuning)
    - n_jobs: Number of jobs.
    - random_state: Random state
    - optimization_direction: Direction of optimization for tuning process
    - n_trials: Number of trials for tuning process
    """
    
    embedding_file_size: int
    min_train_embedding_id: int
    max_train_embedding_id: int
    head_models: list
    models: list
    scoring_metric: str
    swap_labels: bool
    n_folds: int
    n_jobs: int
    random_state: int
    optimization_direction: str
    n_trials: int


@dataclass(frozen=True)
class PredictingConfig:
    """
    Configuration for the prediction process.
    
    Attributes:
    """
    
    embedding_file_size: int
    embedding_ids: list
    head_models: list
    second_level_model: str
    metrics_names: list