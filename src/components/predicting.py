import os
import sys
import math
import shutil
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import *
from tqdm.notebook import tqdm
from sklearn.metrics import get_scorer

from src import logger
from src.models.meta import HeadModel
from src.entity.dataset import CustomDataset4
from src.models.duo import ConvRecDuoModel, MHSARecDuoModel, ISSARecDuoModel, ParallelConvRecDuoModel
from src.models.trio import MHSAParallelConvRecModel, ParallelConvMHSARecModel
from src.models.linear import CustomLinear
from src.models.recurrent import RecModel, ParallelRecModels
from src.models.convolutional import ConvModel, ParallelConvModels
from src.utils.common import save_tensor, load_tensor, clear_vram, read_yaml, load_pkl, save_json
from src.entity.config_entity import PredictingConfig



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)

    

class Predictor:
    """
    Class for making predictions using trained stacking model.

    Attributes:
    - config (PredictingConfig): Configuration settings.
    - path (dict): Dictionary with path values
    """
    
    def __init__(self, config: PredictingConfig, path: dict):
        """
        Initializes the Predictor component.
        
        Args:
        - config (PredictingConfig): Configuration settings.
        - path (dict): Dictionary with path values
        """
        
        self.config = config
        self.path = path
        self.params = {}
        for head_model in self.config.head_models:
            self.params[head_model] = read_yaml(Path(os.path.join(path.path_to_params, f"{head_model}.yaml")))
        

    def read_encoded_data(self, 
                          it: int, 
                          is_new_data: bool=False) -> torch.Tensor:
        """
        Reads a single file with encoded data.

        Parameters:
        - it (int): Id of file with encoded data.

        Returns:
        - data (torch.Tensor): Encoded data.        
        """

        prefix = ''
        if is_new_data: prefix = 'NEW_'

        data = load_tensor(self.path.path_to_encoded_data, f'{prefix}embeddings_{it}.t', logging=False)
        
        return data


    def read_target_data(self, is_new_data: bool=False) -> torch.Tensor:
        """
        Reads target data as tensor.

        Returns:
        - data (torch.Tensor): Tensor with target data.
        """

        prefix = ''
        if is_new_data: prefix = 'NEW_'
            
        data = load_tensor(self.path.path_to_target_data, f'{prefix}target.t')

        # leaving only necessary target data
        groups = [(self.config.embedding_file_size*embedding_id, 
                   self.config.embedding_file_size*(embedding_id+1)) for embedding_id in self.config.embedding_ids]
        indexes = [k for group in groups for k in list(range(*group))]
        indexes = [k for k in indexes if k < len(data)]
        data = data[indexes].detach().numpy()
        
        return data


    def load_head_model(self, head_model: str) -> Module:
        """
        Loads head model.

        Parameters:
        - head_model (str): Name of file with a head model config.

        Returns:
        - model (Module): Head model object.
        """

        model = HeadModel(main_model=str_to_class(self.params[head_model].main_model_class),
                          output_size=self.params[head_model].params['hidden_layers'][-1],
                          n_targets=2,
                          main_model_kwargs=self.params[head_model].params)

        model._modules['main_model'].load_state_dict(torch.load(os.path.join(self.path.path_to_models, f'{self.params[head_model].name}_main_model.pt')))
        model._modules['ffnn'].load_state_dict(torch.load(os.path.join(self.path.path_to_models, f'{self.params[head_model].name}_ffnn.pt')))
        
        model.eval()
        model = model.to('cuda')

        return model


    def load_second_level_model(self):
        """
        Loads second level model 

        Returns:
        - model: Second level model
        """
        
        model = load_pkl(self.path.path_to_models, f'{self.config.second_level_model}.pkl')
        return model
        

    def logits_on_batch(self, 
                        batch_X: torch.Tensor,
                        model: Module) -> torch.Tensor:
        """
        Function to calculate logits for 0 class on a batch of data.

        Parameters:
        - batch_X (torch.Tensor): Tensor with embeddings.
        - model (Module): Model's object.

        Returns:
        - logits (torch.Tensor): Logits for 0 class
        """
        
        with torch.no_grad():
            
            # Calculating logits on the batch
            logits = model(batch_X)

            # Calculating prediction based on logits
            logits = logits[:, 0].to('cpu')
        
        return logits


    def calculate_logits(self, 
                         head_model_name: str,
                         model: Module,
                         is_new_data: bool=False) -> torch.Tensor:
        """
        Function to calculate logits for 0 class using head model on the whole data.

        Parameters:
        - model (Module): Head model.
        - batch_size (int): Batch size.
        
        Returns:
        - logits (torch.Tensor): Logits for 0 class
        """        

        # Creating iterator over embeddings
        iterations = tqdm(enumerate(self.config.embedding_ids),
                          desc='embeddings',
                          leave=True,
                          total=len(self.config.embedding_ids))

        # Creating empty list to store logits
        logits = []

        # Calculating logits for all embeddings
        for it, (embedding_id) in iterations:

            # Reading training data and converting it to CustomDataset3
            bottom = self.config.embedding_file_size*embedding_id
            top = self.config.embedding_file_size*(embedding_id + 1)
            train_dataset = CustomDataset4(X=self.read_encoded_data(embedding_id, is_new_data))

            # Creating generator to yield data in batches
            batch_generator = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                          batch_size=self.params[head_model_name].training_configs.batch_size,
                                                          shuffle=False)


            # Calculating logits for each batch of embeddings
            for batch_X in batch_generator:
    
                # Retrieving logits calculated on that batch
                batch_logits = self.logits_on_batch(batch_X.to('cuda'), model)
    
                # Appending calculated logits to the list with all logits
                logits.append(batch_logits)

            # Deleting current train_dataset
            del train_dataset

        # Merging list with logits to a single Tensor
        logits = torch.cat(logits, dim=0)

        # Returning logits
        return logits


    def make_first_level_predictions(self, is_new_data: bool=False) -> pd.DataFrame:
        
        all_logits = pd.DataFrame()
        
        iterations = tqdm(self.config.head_models,
                        desc='head_models',
                        leave=True,
                        total=len(self.config.head_models))
        iterations.set_postfix({'HEAD_MODEL': None})

        for head_model_name in iterations:

            iterations.set_postfix({'HEAD_MODEL': head_model_name})

            # Loading head model
            model = self.load_head_model(head_model_name)
            
            # Calculating logits
            logits = self.calculate_logits(head_model_name, model, is_new_data)
            logits = logits.detach().numpy()
            all_logits[head_model_name] = logits
            
            # Deleting model
            del model     

            # Clearing VRAM cache
            clear_vram()

            logger.info(f'Logits have been calculated for {head_model_name}')

        return all_logits


    def make_second_level_prediction(self, 
                                     data: pd.DataFrame, 
                                     is_new_data: bool=False):

        model = self.load_second_level_model()
        data['second_level_pred'] = model.predict(data.values)

        for col in data.columns:
            if col != 'second_level_pred':
                data[col] = np.where(data[col] >= 0.5, 0, 1)

        return data


    def evaluate(self, 
                 data: pd.DataFrame, 
                 is_new_data: bool=False):

        prefix = ''
        if is_new_data: prefix='NEW_'

        # Preparing functions to calculate metrics
        metrics = dict(zip(self.config.metrics_names, [get_scorer(metric_name) for metric_name in self.config.metrics_names]))
        
        # Calculating metrics
        values = {}
        for col in [k for k in data.columns if k != 'target']:

            values[col] = {}
            for (metric_name, metric_func) in metrics.items():
                if metric_name in ['precision', 'recall', 'f1']:
                    values[col][metric_name+'_pos'] = metric_func._score_func(y_true=data['target'], y_pred=data[col])
                    values[col][metric_name+'_neg'] = metric_func._score_func(y_true=1-data['target'], y_pred=1-np.array(data[col]))
                else:
                    values[col][metric_name] = metric_func._score_func(y_true=data['target'], y_pred=data[col])   

        # Saving evaluation results
        save_json(Path(self.path.path_to_logs, f'{prefix}metrics_{self.config.second_level_model}.json'), values)


    def save_predictions(self, 
                         data: pd.DataFrame, 
                         is_new_data: bool=False):

        prefix = ''
        if is_new_data: prefix='NEW_'
            
        data.to_csv(os.path.join(self.path.path_to_predictions, f'{prefix}predictions.csv'))
    

    def run_stage(self, 
                  is_new_data: bool=False, 
                  evaluate: bool=False):
        """
        Function to run the stage
        """ 

        logger.info(f'=== STARTING PREDICTION STAGE ===')

        # Making first level predictions
        data = self.make_first_level_predictions(is_new_data)
        logger.info(f'Part1. First level predictions have been calculated')

        # Making second level predictions
        data = self.make_second_level_prediction(data, is_new_data)
        logger.info(f'Part2. Second level prediction has been calculated')

        # Loading target data
        if evaluate:
            data['target'] = self.read_target_data(is_new_data)
            logger.info(f'Part3. Target data has been loaded')
        else:
            logger.info(f'Part3. Skipping target data loading part. Evaluation is not requested')

        # Saving predictions
        self.save_predictions(data, is_new_data)
        logger.info(f'Part4. Predictions have been saved')

        # Evaluating
        if evaluate:
            self.evaluate(data, is_new_data)
            logger.info(f'Part5. Predictions have been evaluated')
        else:
            logger.info(f'Part5. Predictions have not been evaluated. Evaluation is not requested')
        
        logger.info(f'=== FINISHED PREDICTION STAGE ===')