import warnings

import torch
from torch import nn 
from transformers import AutoModel

from src.models.linear import CustomLinear
from src.models.recurrent import RecModel, ParallelRecModels
from src.models.convolutional import ConvModel, ParallelConvModels



class ValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)



class MetaModel(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 embedding_model_checkpoint: str, 
                 main_model, 
                 output_size: int, 
                 n_targets: int, 
                 main_model_kwargs: dict):
        super(MetaModel, self).__init__()
        
        self.add_module('embedding_model', AutoModel.from_pretrained(embedding_model_checkpoint, num_labels=n_targets))
        for param in self.embedding_model.base_model.parameters():
            param.requires_grad = False
        self.add_module('main_model', main_model(**main_model_kwargs))
        self.add_module('ffnn', nn.Sequential())
        self._modules['ffnn'].add_module('classifier', nn.Linear(output_size, n_targets))
        self._modules['ffnn'].add_module('softmax', nn.Softmax(dim=1))

    
    def forward(self, input_ids, attention_mask):
        x = self._modules['embedding_model'](input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        x = self._modules['main_model'](x)
        x = self._modules['ffnn'](x)
        return x



class HeadModel(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 main_model, 
                 output_size: int, 
                 n_targets: int, 
                 main_model_kwargs: dict):
        super(HeadModel, self).__init__()
        
        self.add_module('main_model', main_model(**main_model_kwargs))
        self.add_module('ffnn', nn.Sequential())
        self._modules['ffnn'].add_module('classifier', nn.Linear(output_size, n_targets))
        self._modules['ffnn'].add_module('softmax', nn.Softmax(dim=1))

    
    def forward(self, x):
        x = self._modules['main_model'](x)
        x = self._modules['ffnn'](x)
        return x