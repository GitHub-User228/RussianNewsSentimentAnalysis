import os
import math
import torch
import pandas as pd
from pandas import DataFrame
from tqdm.notebook import tqdm
from transformers import AutoModel

from src import logger
from src.entity.dataset import CustomDataset2
from src.utils.common import load_tensor, save_tensor
from src.entity.config_entity import DataEncodingConfig



class DataEncoding:
    """
    Class for encoding tokenized data.

    Attributes:
    - config (DataEncodingConfig): Configuration settings.
    - path (dict): Dictionary with path values
    """
    
    def __init__(self, config: DataEncodingConfig, path: dict):
        """
        Initializes the DataEncoding component.
        
        Args:
        - config (DataEncodingConfig): Configuration settings.
        - path (dict): Dictionary with path values
        """
        self.config = config
        self.path = path


    def read_data(self, is_new_data: bool = False) -> dict:
        """
        Reads tokenized data as tensors.

        Parameters:
        - is_new_data (bool): Whether data to be read is new.

        Returns:
        - data (dict): Dictionary with tokenized data.
        """

        prefix = ''
        if is_new_data: prefix = 'NEW_'
        data = {}
        data['input_ids'] = load_tensor(self.path.path_to_tokenized_data, f'{prefix}input_ids.t')
        data['attention_mask'] = load_tensor(self.path.path_to_tokenized_data, f'{prefix}attention_mask.t')
        
        return data


    def load_model(self, device='cuda'):
        """
        Function to load a Hugging Face model and move it to GPU

        Returns
        - model (object): Model object.
        """
        model = AutoModel.from_pretrained(self.config.model_checkpoint, num_labels=self.config.n_labels).to(device)
        return model
        

    def encode_data(self,
                    data: CustomDataset2, 
                    is_new_data: bool = False):
        """
        Function to encode tokenized data and save resulting embeddings

        Parameters:
        - data (CustomDataset2): Tokenized data as CustomDataset2.
        - is_new_data (bool): Whether input data is new.
        """
        
        prefix=''
        if is_new_data:
            prefix='NEW_'

        # Loading model
        model = self.load_model()
        model.eval()
        logger.info('1. Model has been loaded')

        # Creating batch generator and tqdm iterator
        batch_generator = torch.utils.data.DataLoader(dataset=data, batch_size=self.config.batch_size, shuffle=False)
        n_batches = math.ceil(len(data)/batch_generator.batch_size)
        iterator = tqdm(enumerate(batch_generator), desc='batch', leave=True, total=n_batches)
        logger.info('2. Batch generator and iterator have been initialized')

        if self.config.start_batch == None: 
            start_batch = 0
        else:
            start_batch = self.config.start_batch
        if self.config.end_batch == None: 
            end_batch = n_batches
        else:
            end_batch = self.config.end_batch

        # Encoding data
        with torch.no_grad():

            s = 0
            k = 0
            output = None
            
            for it, (batch_ids, batch_masks) in iterator:
                    
                    if (it >= start_batch) and (it <= end_batch):

                        # Moving tensors to GPU
                        batch_ids = batch_ids.to('cuda')
                        batch_masks = batch_masks.to('cuda')
            
                        # Getting embeddings
                        batch_output = model(input_ids=batch_ids, attention_mask=batch_masks).last_hidden_state.to('cpu').to(torch.float32)
        
                        # Merging outputs
                        k += 1
                        output = batch_output if output==None else torch.cat([output, batch_output], axis=0)
            
                        # Saving embeddings
                        if (k == self.config.n_batches_per_save) or (it >= n_batches - 1):
                            save_tensor(self.path.path_to_encoded_data, f'{prefix}embeddings_{s}.t', output)
                            s += 1
                            k = 0
                            output = None

                    else:
                        
                        # Updating counters
                        k += 1
                        if (k == self.config.n_batches_per_save) or (it >= n_batches - 1):
                            s += 1
                            k = 0
                            output = None
                        
            
        logger.info('3. Data has been encoded and saved')

        # Clearing cache
        torch.cuda.empty_cache()
        del model
        del batch_generator
        del data
        del output
        del batch_ids
        del batch_masks
        del batch_output
        logger.info('4. Cache has been cleared')


    def run_stage(self, is_new_data: bool = False):
        """
        Function to encode tokenized data and save resulting embeddings.

        Parameters:
        - is_new_data (bool): Whether data to be encoded is new.
        """

        logger.info('=== STARTING ENCODING STAGE ===')
            
        # Loading preprocessed data
        data = self.read_data(is_new_data)
        logger.info('Part1. Tokenized and target data has been loaded')

        # Converting data to CustomDataset form
        data = CustomDataset2(**data)
        logger.info('Part2. Data has been converted to Dataset form')

        # Encoding data
        self.encode_data(data, is_new_data)
        logger.info('Part3. Data has been encoded and saved')
        
        logger.info('=== FINISHED ENCODING STAGE ===')