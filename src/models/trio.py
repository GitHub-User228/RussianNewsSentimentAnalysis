import warnings

import torch
from torch import nn 

from src import logger
from src.models.linear import LinearBlock
from src.models.recurrent import RecModel
from src.models.convolutional import ConvModel, ParallelConvModels
from src.models.attention import MultiHeadSelfAttention, InterlacedSparseMHSA, InterlacedSparseSA



class ValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)



class MHSAParallelConvRecModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 mhsa_model_params: dict,
                 parallel_conv_model_params: dict,
                 conv_output_groups: dict,
                 rec_models_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(MHSAParallelConvRecModel, self).__init__()

        self.add_module('mhsa', MultiHeadSelfAttention(**mhsa_model_params))

        self.conv_output_groups = conv_output_groups

        if len(parallel_conv_model_params['hidden_layers']) > 0:
            raise ValidationError('ParallelConvModels must has no values specified for hidden_layers')    

        if any([len(conv_model['blocks_kwargs']) > 1 for conv_model in parallel_conv_model_params['conv_models_kwargs'].values()]):
            raise ValidationError('ParallelConvModels must has only one conv_block specified for each parallel conv_model')  
        
        self.add_module('parallel_conv_models', ParallelConvModels(**parallel_conv_model_params))

        for (k, v) in rec_models_params.items():
            
            name = list(v['rec_model_kwargs'].keys())[0]
            
            if v['rec_model_kwargs'][name]['input_size'] != None:
                raise ValidationError('RecModel must has no value specified for input_size. It is calculated automatically')

            if len(v['hidden_layers']) > 0:
                raise ValidationError('RecModel must has no values specified for hidden_layers')

        # Calculating output number of features for each group of conv models
        outputs_n_features = {}
        for (k, v) in conv_output_groups.items():
            outputs_n_features[k] = sum([self._modules['parallel_conv_models']._modules['convolutional_models'][k2].output_size['features'] for k2 in v])
        
        self.add_module('recurrent_models', nn.ModuleDict())
        for (k, v) in rec_models_params.items():
            rec_models_params[k]['rec_model_kwargs'][name]['input_size'] = outputs_n_features[k]
            if rec_models_params[k]['rec_model_kwargs'][name]['hidden_size'] == None:
                logger.info('WARNING: One of RecModels has no value specified for hidden_size. Using input_size value')
                rec_models_params[k]['rec_model_kwargs'][name]['hidden_size'] = rec_models_params[k]['rec_model_kwargs'][name]['input_size']
            self._modules['recurrent_models'].update({k: RecModel(**rec_models_params[k])})

        self.add_module('ffnn', nn.Sequential())

        input_dim = 0
        for (k, v) in rec_models_params.items():
            input_dim_curr = rec_models_params[k]['rec_model_kwargs'][name]['hidden_size']
            if rec_models_params[k]['rec_model_kwargs'][name]['bidirectional']:
                input_dim_curr = input_dim_curr * 2
            input_dim += input_dim_curr
                
        for it, hidden_dim in enumerate(hidden_layers):
            self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
            input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules['mhsa'](x)
        x = self._modules['parallel_conv_models'](x)
        output = []
        for k in self._modules['recurrent_models'].keys():
            output.append(self._modules['recurrent_models'][k](torch.cat([x[k2] for k2 in self.conv_output_groups[k]], axis=-1)))
        output = torch.cat(output, axis=-1)
        output = self._modules['ffnn'](output)
        return output    



class ParallelConvMHSARecModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 mhsa_model_params: dict,
                 parallel_conv_model_params: dict,
                 conv_output_groups: dict,
                 rec_models_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(ParallelConvMHSARecModel, self).__init__()

        self.conv_output_groups = conv_output_groups

        if len(parallel_conv_model_params['hidden_layers']) > 0:
            raise ValidationError('ParallelConvModels must has no values specified for hidden_layers')    

        if any([len(conv_model['blocks_kwargs']) > 1 for conv_model in parallel_conv_model_params['conv_models_kwargs'].values()]):
            raise ValidationError('ParallelConvModels must has only one conv_block specified for each parallel conv_model')  
        
        self.add_module('parallel_conv_models', ParallelConvModels(**parallel_conv_model_params))

        self.add_module('mhsa_models', nn.ModuleDict())
        for (k, v) in conv_output_groups.items():
            self._modules['mhsa_models'].add_module(f'mhsa_{k}', MultiHeadSelfAttention(**mhsa_model_params))

        for (k, v) in rec_models_params.items():
            
            name = list(v['rec_model_kwargs'].keys())[0]
            
            if v['rec_model_kwargs'][name]['input_size'] != None:
                raise ValidationError('RecModel must has no value specified for input_size. It is calculated automatically')

            if len(v['hidden_layers']) > 0:
                raise ValidationError('RecModel must has no values specified for hidden_layers')

        # Calculating output number of features for each group of conv models
        outputs_n_features = {}
        for (k, v) in conv_output_groups.items():
            outputs_n_features[k] = sum([self._modules['parallel_conv_models']._modules['convolutional_models'][k2].output_size['features'] for k2 in v])
        
        self.add_module('recurrent_models', nn.ModuleDict())
        for (k, v) in rec_models_params.items():
            rec_models_params[k]['rec_model_kwargs'][name]['input_size'] = outputs_n_features[k]
            if rec_models_params[k]['rec_model_kwargs'][name]['hidden_size'] == None:
                logger.info('WARNING: One of RecModels has no value specified for hidden_size. Using input_size value')
                rec_models_params[k]['rec_model_kwargs'][name]['hidden_size'] = rec_models_params[k]['rec_model_kwargs'][name]['input_size']
            self._modules['recurrent_models'].update({k: RecModel(**rec_models_params[k])})

        self.add_module('ffnn', nn.Sequential())

        input_dim = 0
        for (k, v) in rec_models_params.items():
            input_dim_curr = rec_models_params[k]['rec_model_kwargs'][name]['hidden_size']
            if rec_models_params[k]['rec_model_kwargs'][name]['bidirectional']:
                input_dim_curr = input_dim_curr * 2
            input_dim += input_dim_curr
                
        for it, hidden_dim in enumerate(hidden_layers):
            self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
            input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules['parallel_conv_models'](x)
        output = []
        for k in self._modules['recurrent_models'].keys():
            x2 = torch.cat([x[k2] for k2 in self.conv_output_groups[k]], axis=-1)
            x2 = self._modules['mhsa_models'][f'mhsa_{k}'](x2)
            output.append(self._modules['recurrent_models'][k](x2))
        output = torch.cat(output, axis=-1)
        output = self._modules['ffnn'](output)
        return output   