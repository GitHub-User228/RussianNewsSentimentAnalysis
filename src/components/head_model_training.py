import os
import sys
import math
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.nn import *
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import *

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm.notebook import tqdm

from src import logger
from src.models.meta import HeadModel
from src.utils.callback import Callback
from src.entity.dataset import CustomDataset3
from src.models.duo import ConvRecDuoModel, MHSARecDuoModel, ISSARecDuoModel, ParallelConvRecDuoModel
from src.models.trio import MHSAParallelConvRecModel, ParallelConvMHSARecModel
from src.models.linear import CustomLinear
from src.models.attention import MultiHeadSelfAttention, InterlacedSparseSA
from src.models.recurrent import RecModel, ParallelRecModels
from src.models.convolutional import ConvModel, ParallelConvModels
from src.utils.common import load_tensor, clear_vram, read_yaml
from src.entity.config_entity import HeadModelTrainingConfig



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)

    

class HeadModelTraining:
    """
    Class for head model training.
    """
    
    def __init__(self, 
                 config: HeadModelTrainingConfig,
                 path: dict, 
                 head_model_filename: str = None,
                 use_default_training_configs: bool = True):
        """
        Initializes the HeadModelTraining component.
        
        Args:
        - config (HeadModelTrainingConfig): Configuration settings.
        - path (dict): Dictionary with path values
        - head_model_filename (str): Name of the file with parameters of a head model to be considered
        - use_default_training_configs (bool). Whether to use default training configs.
        """
        
        self.config = config
        params = read_yaml(Path(os.path.join(path.path_to_params, f"{head_model_filename}.yaml")))
        self.head_model_name = params['name']
        self.main_model_class = params['main_model_class']
        self.head_model_params = params['params']
        if use_default_training_configs:
            self.training_configs = self.config.training_configs
        else:
            self.training_configs = params['training_configs']
        self.callback_logdir = params['callback_logdir']
        self.path = path


    def read_encoded_data(self, it: int) -> torch.Tensor:
        """
        Reads file with a part of encoded data.

        Parameters:
        - it (int): ID of file with encoded data.

        Returns:
        - data (torch.Tensor): Tensor with encoded data.
        """

        data = load_tensor(self.path.path_to_encoded_data, f'embeddings_{it}.t', logging=False)
        
        return data


    def read_target_data(self) -> torch.Tensor:
        """
        Reads target data as tensor.

        Returns:
        - data (torch.Tensor): Tensor with target data.
        """
        
        data = load_tensor(self.path.path_to_target_data, 'target.t')
        
        return data


    def train_on_batch(self, 
                       batch_X: torch.Tensor, 
                       batch_y: torch.Tensor, 
                       model: Module) -> (float, torch.Tensor):
        """
        Function to make training step on a batch of data.

        Parameters:
        - batch_X (torch.Tensor): Tensor with embeddings.
        - batch_y (torch.Tensor): Tensor with target data.
        - model (Module): Model's object.

        Returns:
        - batch_loss (float): Loss value on the batch.
        - pred (torch.Tensor): Predictions on the batch
        """

        # Changing model's mode to training mode
        model.train()

        # Making gradients equal to 0
        model.zero_grad()

        # Calculating logits on the batch
        logits = model(batch_X)

        # Calculating loss on the batch
        loss = self.loss_function(logits, batch_y.to(torch.int64))

        # Making backward step
        loss.backward()

        # Making optimizer step
        self.optimizer.step()

        # Making scheduler step (if scheduler is specified)
        if self.scheduler != None:
            self.scheduler.step()

        # Extracting batch loss value
        batch_loss = loss.cpu().item()

        # Calculating prediction based on logits
        pred = logits.argmax(dim=-1).to('cpu')
        
        return batch_loss, pred


    def train_on_epoch(self, 
                       target_data: torch.Tensor,
                       model: Module) -> float:
        """
        Function to train model during one epoch.

        Parameters:
        - target_data (torch.Tensor): Tensor with target data.
        - model (Module): Model's object.
        
        Returns:
        - epoch_loss (float): Loss value on the epoch.
        """        

        # Creating variables to store epoch loss and count processed size of data
        epoch_loss = 0
        total = 0

        # Creating iterator over embeddings
        iterations = tqdm(enumerate(range(self.config.min_train_embedding_id, self.config.max_train_embedding_id+1)),
                          desc='embeddings',
                          leave=True,
                          total=self.config.max_train_embedding_id-self.config.min_train_embedding_id+1)
        iterations.set_postfix({'train batch loss': np.nan, 'lr': np.nan})

        # Performing training process for all embeddings in epoch
        for it, (embedding_id) in iterations:

            # Reading training data and converting it to CustomDataset3
            bottom = self.config.embedding_file_size*embedding_id
            top = self.config.embedding_file_size*(embedding_id + 1)
            train_dataset = CustomDataset3(X=self.read_encoded_data(embedding_id), y=target_data[bottom:top])

            # Creating generator to yield data in batches
            batch_generator = torch.utils \
                                   .data \
                                   .DataLoader(dataset=train_dataset, 
                                               batch_size=self.training_configs.batch_size,
                                               shuffle=True,
                                               generator=torch.Generator().manual_seed(self.training_configs.random_state))

            # Performing training process for each batch of embeddings
            for (batch_X, batch_y) in batch_generator:
    
                # Training on a batch and retrieving loss value calculated on that batch
                batch_loss, batch_pred = self.train_on_batch(batch_X.to('cuda'), batch_y.to('cuda'), model)
    
                # Updating callback with new batch loss and current learning rate
                current_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                if self.callback is not None:
                    model.eval()
                    self.callback(model, batch_loss, batch_pred, batch_y.to('cpu'))
    
                # Updating tqdm postfix with new batch loss and current learning rate
                iterations.set_postfix({'train batch loss': batch_loss, 'lr': current_lr})
    
                # Updating variables (then to be used to calculate loss on epoch)
                epoch_loss += batch_loss*len(batch_y)
                total += len(batch_y)

            # Deleting current train_dataset
            del train_dataset

        # Calculating and returning epoch loss value
        epoch_loss = epoch_loss/total
        return epoch_loss


    def fit(self,
            target_data: torch.Tensor,
            model: Module,
            n_epochs: int = None):
        """
        Function to train head model.

        Parameters:
        - target_data (torch.Tensor): Tensor with targer data.
        - model (Module): Model's object.
        - n_epochs (int): Number of epochs. If set, this value will be used instead of one specified in configs.
        """ 

        # Initialization of optimizer, loss function and scheduler
        self.loss_function = str_to_class(self.training_configs.loss_function)()
        self.optimizer = str_to_class(self.training_configs.optimizer)(model.parameters(), lr=self.training_configs.learning_rate)
        self.scheduler = None
        if self.training_configs.scheduler != None:
            self.scheduler = str_to_class(self.training_configs.scheduler)(optimizer=self.optimizer)

        # Initializationg of callback (if specified)
        self.callback = None
        if self.training_configs.use_callback:

            log_dir = os.path.join(self.path.path_to_logs, self.callback_logdir)

            if self.training_configs.overwrite_existing_callback and os.path.exists(log_dir):
                shutil.rmtree(log_dir)

            writer = SummaryWriter(log_dir = log_dir)
            
            bottom = self.config.embedding_file_size*(self.config.validation_embedding_id)
            top = self.config.embedding_file_size*(self.config.validation_embedding_id+1)
            self.callback = Callback(writer=writer, 
                                     dataset=CustomDataset3(X=self.read_encoded_data(self.config.validation_embedding_id), 
                                                            y=target_data[bottom:top]),
                                     loss_function=str_to_class(self.training_configs.loss_function)(),
                                     loss_function_name=self.training_configs.loss_function,
                                     batch_size=self.training_configs.eval_batch_size,
                                     eval_steps=self.training_configs.eval_steps,
                                     metrics_names=self.training_configs.metrics_names)

        # Creating iterator over epochs
        if type(n_epochs) == int:
            if n_epochs > 0:
                logger.info('WARNING: Using number of epochs specified in arguments instead of one in configs.')
                iterations = tqdm(range(n_epochs), desc='epochs', leave=True)
            else:
                raise ValueError('Number of epochs should be more than 0')
        else:
            iterations = tqdm(range(self.training_configs.n_epochs), desc='epochs', leave=True)
            n_epochs = self.training_configs.n_epochs
        iterations.set_postfix({'train epoch loss': np.nan})

        # Performing training process for each epoch
        for it in iterations:
            
            # Performing training process for each batch in epoch
            epoch_loss = self.train_on_epoch(target_data=target_data, model=model)

            # Updating tqdm postfix
            iterations.set_postfix({'train epoch loss': epoch_loss})

            # Checking if need to make a checkpoint
            if self.training_configs.checkpoint_each_epoch or (it == n_epochs - 1):
                torch.save(model._modules['main_model'].state_dict(), 
                           os.path.join(self.path.path_to_models, self.head_model_name+'_main_model.pt'))
                torch.save(model._modules['ffnn'].state_dict(), 
                           os.path.join(self.path.path_to_models, self.head_model_name+'_ffnn.pt'))

        # Deleting model
        del model

        # Clearing cache
        clear_vram()


    def run_stage(self, 
                  n_epochs: int = None):
        """
        Function to run head model training stage

        Parameters:
        - n_epochs (int): Number of epochs. If set, this value will be used instead of one specified in configs.
        """ 

        logger.info(f'=== STARTING TRAINING STAGE for {self.head_model_name} ===')
        
        # Reading target data
        target_data = self.read_target_data()
        logger.info('Part1. Target data has been loaded')
        
        # Initialization of main model
        model = HeadModel(main_model=str_to_class(self.main_model_class),
                          output_size=self.head_model_params['hidden_layers'][-1],
                          n_targets=2,
                          main_model_kwargs=self.head_model_params).to('cuda')
        logger.info('Part2. Head model has been initialized')
        
        # Training
        self.fit(target_data, model, n_epochs)
        logger.info('Part3. Training has been completed')

        logger.info(f'=== FINISHED TRAINING STAGE for {self.head_model_name} ===')