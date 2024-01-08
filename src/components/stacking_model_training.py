import os
import sys
from pathlib import Path

import torch
import optuna
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from optuna.samplers import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src import logger
from src.utils.common import load_tensor, read_yaml, save_json, save_pkl
from src.entity.config_entity import StackingModelTrainingConfig



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)

    

class StackingModelTraining:
    """
    Class for training a second level model of a stacking model.
    """
    
    def __init__(self, 
                 config: StackingModelTrainingConfig,
                 path: dict,
                 models: list=None):
        """
        Initializes the StackingModelTraining component.
        
        Args:
        - config (StackingModelTrainingConfig): Configuration settings.
        - path (dict): Dictionary with path values
        - models (list): List with names of yaml files with second level models configs. 
                         If None, data from configs will be used.
        """
        
        self.config = config
        self.path = path
        self.models = models
        if models == None:
            self.models = self.config.models
        self.params = {}
        for model in self.models:
            self.params[model] = read_yaml(Path(os.path.join(path.path_to_params, f"{model}.yaml")))


    def read_target_data(self) -> np.ndarray:
        """
        Reads target data (which corresponds to specified ids in configs) and returns it as np.ndarray.

        Returns:
        - data (np.ndarray): Numpy array with target data.
        """

        # Loading target data
        data = load_tensor(self.path.path_to_target_data, 'target.t')

        # leaving only necessary target data
        ids = list(range(self.config.min_train_embedding_id, self.config.max_train_embedding_id+1))
        groups = [(self.config.embedding_file_size*embedding_id, self.config.embedding_file_size*(embedding_id+1)) for embedding_id in ids]
        indexes = [k for group in groups for k in list(range(*group))]
        indexes = [k for k in indexes if k < len(data)]
        data = data[indexes].detach().numpy()

        # Returning necessary target data
        return data


    def read_first_level_logits(self) -> pd.DataFrame:
        """
        Reads first level predictions (logits) for all folds and returns them as pd.DataFrame.

        Returns:
        - data (pd.DataFrame): Pandas DataFrame with first level logits.
        """

        # Reading logits for each specified head model in configs
        data = pd.DataFrame()
        for head_model_name in self.config.head_models:
            logits = []
            # Reading logits for each fold
            for file in sorted([file for file in os.listdir(self.path.path_to_logits) if f'{head_model_name}_' in file]):
                logits.append(load_tensor(self.path.path_to_logits, file, logging=False))
            data[head_model_name] = torch.cat(logits, dim=0).detach().numpy()

        # Returning data with logits
        return data
        

    def objective(self, trial):
        """
        Objective function to be utilized by Optuna during tuning process.
        Firstly, a sample for specified hyperparameters is made.
        Secondly, mean cross-validation value for specified metric in configs is calculated for model to be tuned.

        Parameters:
        - trial: 

        Returns:
        - mean_score (float): Mean cross-validation score.
        """

        # Getting a sample of hyperparameters
        hyperparameters = {}
        for (hyperparameter, config) in self.params[self.current_model].grid.items():
            if config.sampling_method == 'suggest_loguniform':
                hyperparameters[hyperparameter] = trial.suggest_loguniform(hyperparameter, **config.params)
            if config.sampling_method == 'suggest_float':
                hyperparameters[hyperparameter] = trial.suggest_float(hyperparameter, **config.params)
            if config.sampling_method == 'suggest_categorical':
                hyperparameters[hyperparameter] = trial.suggest_categorical(hyperparameter, **config.params)
            if config.sampling_method == 'suggest_int':
                hyperparameters[hyperparameter] = trial.suggest_int(hyperparameter, **config.params)

        # Initialization of a model
        model = str_to_class(self.params[self.current_model].model)(**self.params[self.current_model].default, **hyperparameters)

        # Swapping labels (in case of metric, that needs to be calculated for 0 class)
        y = self.data['target']
        if self.config.swap_labels:
            y = 1 - y
            
        # Cross-Validating
        scores = cross_val_score(estimator=model, 
                                 X=self.data[[k for k in self.data.columns if k!='target']], 
                                 y=y, 
                                 scoring=self.config.scoring_metric,
                                 cv=self.config.n_folds,
                                 n_jobs=self.config.n_jobs)
        mean_score = np.mean(scores)

        # Returning mean score
        return mean_score


    def tune(self):
        """
        Tunes specified second level models using Optuna.
        Saves evaluation results for the best trial as json.
        """

        # Initializationg of a iterator over second level models
        iterator = tqdm(self.models, desc='Tuning Models')
        iterator.set_postfix({'Model': None})

        # Tuning each second level model
        for model_name in iterator:
            
            iterator.set_postfix({'Model': model_name})

            # Updating current model name (to be used inside objective function)
            self.current_model = model_name

            # Initialization of a sampler
            if self.params[self.current_model].sampler == 'GridSampler':
                search_space = {}
                for (hyperparameter, config) in self.params[self.current_model].grid.items():
                    search_space[hyperparameter] = config.params.choices
                sampler = GridSampler(search_space=search_space, seed=self.config.random_state)
            if self.params[self.current_model].sampler == 'TPESampler':
                sampler = TPESampler(seed=self.config.random_state)

            # Initialization of a study (tuner)
            study = optuna.create_study(sampler=sampler, direction=self.config.optimization_direction)

            # Tuning
            study.optimize(self.objective, n_trials=self.config.n_trials)

            # Saving best results and best model
            self.save_best_results(study, model_name)
            self.save_best_model(study, model_name)
            
            logger.info(f'Model {model_name} has been tuned')
            
        
    def save_best_results(self, study, model_name: str):
        """
        Saves results for the best trial of the tuning process 

        Parameters:
        - study: Completed study of the tuning process.
        - model_name (str): Name of the second level model
        """
        
        results = {}
        results[self.config.scoring_metric] = study.best_trial.value
        results['best_trial_number'] = study.best_trial.number
        results['best_hyperparameters'] = study.best_params
        
        save_json(Path(os.path.join(self.path.path_to_logs, f'{model_name}.json')), results)


    def save_best_model(self, study, model_name: str):
        """
        Saves second level model for the best trial of the tuning process 

        Parameters:
        - study: Completed study of the tuning process.
        - model_name (str): Name of the second level model
        """

        model = str_to_class(self.params[self.current_model].model)(**self.params[self.current_model].default, **study.best_params)
        model = model.fit(self.data[[k for k in self.data.columns if k!='target']], self.data['target'])
        save_pkl(model, self.path.path_to_models, f'{model_name}.pkl')
        

    def run_stage(self):
        """
        Function to run corresponding stage
        """ 

        logger.info(f'=== STARTING TRAINING STAGE for stacking models ===')
        
        # Reading target data
        target_data = self.read_target_data()
        logger.info('Part1. Target data has been loaded')

        # Reading data with first level predictions 
        self.data = self.read_first_level_logits()
        self.data['target'] = target_data
        logger.info('Part2. First level logits have been loaded')
        
        # Tuning
        self.tune()
        logger.info('Part3. All specified second level models have been tuned')
        
        logger.info(f'=== FINISHED TRAINING STAGE for stacking models ===')