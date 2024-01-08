import os
import math
import torch
import pandas as pd
from pandas import DataFrame
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModel

from src import logger
from src.utils.common import load_tensor, save_tensor
from src.entity.config_entity import DataTokenizationConfig



class DataTokenization:
    """
    Class for text tokenization based on selected tokenizer.
    Tokenized data is saved separately.

    Attributes:
    - config (DataTokenizationConfig): Configuration settings.
    - path (dict): Dictionary with path values
    """
    
    def __init__(self, config: DataTokenizationConfig, path: dict):
        """
        Initializes the DataTokenization component.
        
        Args:
        - config (DataTokenizationConfig): Configuration settings.
        - path (dict): Dictionary with path values
        """
        self.config = config
        self.path = path


    def read_preprocessed_data(self, is_new_data: bool=False) -> DataFrame:
        """
        Reads the preprocessed data as csv.

        Parameters:
        - is_new_data (bool): Whether data to be read is new.

        Returns:
        - df (DataFrame): Pandas DataFrame containing the read data.
        """
        try:
            prefix = ''
            if is_new_data:
                prefix = 'NEW_'
            df = pd.read_csv(os.path.join(self.path.path_to_preprocessed_data, prefix+'data.csv')).dropna(axis=0)
            return df
        except Exception as e:
            logger.error(f"Failed to read data. Error: {e}")
            raise e


    def read_tokenized_data(self) -> dict:
        """
        Reads tokenized data (input_ids and attention_mask tensors).

        Returns:
        - df (dict): Dictionary with input_ids and attention_mask tensors as torch.Tensor each.
        """
        try:
            df = {}
            df['input_ids'] = load_tensor(self.path.path_to_tokenized_data, 'input_ids.t')
            df['attention_mask'] = load_tensor(self.path.path_to_tokenized_data, 'attention_mask.t')
            return df
        except Exception as e:
            logger.error(f"Failed to read data. Error: {e}")
            raise e


    def load_tokenizer(self):
        """
        Function to load a Hugging Face tokenizer

        Returns:
        - tokenizer (object): Tokenizer object.
        """
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint, use_fast=False)
        tokenizer.pad_token = '<pad>'
        return tokenizer
        

    def tokenize_data(self, 
                      data: DataFrame, 
                      is_new_data: bool = False) -> dict:
        """
        Function to tokenize preprocessed text data and save it

        Parameters:
        - data (DataFrame): Preprocessed text data as pandas DataFrame
        - is_new_data (bool): Whether input data is new.

        Returns:
        - data (dict): Tokenized data as dict with input_ids and attention_masks.
        """

        prefix = ''
        if is_new_data:
            prefix = 'NEW_'

        # Checking if target data exists
        if prefix+'target.t' in os.listdir(self.path.path_to_target_data):
            # Loading target
            target = load_tensor(self.path.path_to_target_data, prefix+'target.t')
            logger.info('1. Target data has been loaded')
        else:
            # Saving target
            target = torch.tensor(data['label'].values).to(torch.int64)
            save_tensor(self.path.path_to_target_data, prefix+'target.t', target)   
            logger.info('1. Target data has been extracted and saved')

        # Checking if tokenized data exists (only for training data)
        if (not is_new_data) and all([f'{name}.t' in os.listdir(self.path.path_to_tokenized_data) for name in ['input_ids', 'attention_mask']]):

            # Loading tokenized data
            data = self.read_tokenized_data()
            logger.info('2-4. Tokenized data has been loaded')

        else:

            # Loading tokenizer
            tokenizer = self.load_tokenizer()
            logger.info('2. Tokenizer has been loaded')
                                  
            # Tokenizing data
            data = tokenizer(list(data['text'].values), padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt')
            logger.info('3. Data has been tokenized')

            # Saving tokenized data (only for training data)
            if not is_new_data:
                save_tensor(self.path.path_to_tokenized_data, 'input_ids.t', data['input_ids'])
                save_tensor(self.path.path_to_tokenized_data, 'attention_mask.t', data['attention_mask'])
                logger.info('4. Tokenized data has been saved')
            else:
                logger.info('4. Tokenized data has not been saved, because it is new data')

        return data


    def run_stage(self, 
                  is_new_data: bool = False, 
                  return_tokenized_data: bool = False) -> dict:
        """
        Function to tokenize preprocessed text data and save resulting data

        Parameters:
        - is_new_data (bool): Whether preprocessed data to be processed is new.
        - return_tokenized_data (bool): Whether to return tokenized data
        """

        logger.info('=== STARTING TOKENIZING STAGE ===')
            
        # Loading preprocessed data
        logger.info('Part1. Reading preprocessed data')
        data = self.read_preprocessed_data(is_new_data)
        logger.info('Part1. Data has been loaded')

        # Tokenizing data
        logger.info('Part2. Tokenizing preprocessed data')
        data = self.tokenize_data(data, is_new_data)
        logger.info('Part2. Data has been tokenized and saved')

        logger.info('=== FINISHED TOKENIZING STAGE ===')
        
        if return_tokenized_data:
            return data
        else:
            del data