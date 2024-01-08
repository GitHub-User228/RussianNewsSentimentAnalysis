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



class ConvRecDuoModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 conv_model_params: dict,
                 rec_model_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(ConvRecDuoModel, self).__init__()

        if len(conv_model_params['hidden_layers']) > 0:
            raise ValidationError('ConvModel must has no values specified for hidden_layers. Otherwise, RecModel can not be applied to the output of ConvModel')
        
        self.add_module('conv_model', ConvModel(**conv_model_params))

        if (len(conv_model_params['blocks_kwargs']) > 1) and (sum(self._modules['conv_model'].return_feature_map.values()) > 1):
            raise ValidationError('ConvModel must return feature map only for the last block.')

        name = list(rec_model_params['rec_model_kwargs'].keys())[0]
        if rec_model_params['rec_model_kwargs'][name]['input_size'] != None:
            raise ValidationError('RecModel must has no value specified for input_size. It is calculated automatically')

        if len(rec_model_params['hidden_layers']) > 0:
            raise ValidationError('RecModel must has no values specified for hidden_layers. Specify them for ConvRecDuoModel')

        if len(hidden_layers) == 0:
            raise ValidationError('ConvRecDuoModel must has at least one value specified for hidden_layers')        
        
        rec_model_params['rec_model_kwargs'][name]['input_size'] = self._modules['conv_model'].output_dim['n_features']
        if rec_model_params['rec_model_kwargs'][name]['hidden_size'] == None:
            logger.info('WARNING: RecModel has no value specified for hidden_size. Using input_size value')
            rec_model_params['rec_model_kwargs'][name]['hidden_size'] = rec_model_params['rec_model_kwargs'][name]['input_size']
        
        self.add_module('rec_model', RecModel(**rec_model_params))

        self.add_module('ffnn', nn.Sequential())
            
        input_dim = rec_model_params['rec_model_kwargs'][name]['hidden_size']
        if rec_model_params['rec_model_kwargs'][name]['bidirectional']:
            input_dim = input_dim * 2
                
        for it, hidden_dim in enumerate(hidden_layers):
            self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
            input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules['conv_model'](x)
        x = self._modules['rec_model'](x)
        x = self._modules['ffnn'](x)
        return x    



class ParallelConvRecDuoModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 parallel_conv_model_params: dict,
                 conv_output_groups: dict,
                 rec_models_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(ParallelConvRecDuoModel, self).__init__()

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

        # if len(hidden_layers) == 0:
        #     raise ValidationError('ParallelConvRecDuoModel must has at least one value specified for hidden_layers') 

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
            output.append(self._modules['recurrent_models'][k](torch.cat([x[k2] for k2 in self.conv_output_groups[k]], axis=-1)))
        output = torch.cat(output, axis=-1)
        output = self._modules['ffnn'](output)
        return output    



class MHSARecDuoModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 mhsa_model_params: dict,
                 rec_model_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(MHSARecDuoModel, self).__init__()

        self.add_module('mhsa', MultiHeadSelfAttention(**mhsa_model_params))

        name = list(rec_model_params['rec_model_kwargs'].keys())[0]
        if rec_model_params['rec_model_kwargs'][name]['input_size'] != None:
            raise ValidationError('RecModel must have no value specified for input_size. It is calculated automatically')
        else:
            rec_model_params['rec_model_kwargs'][name]['input_size'] = mhsa_model_params.n_features

        if len(rec_model_params['hidden_layers']) > 0:
            raise ValidationError('RecModel must have no values specified for hidden_layers. Specify them for MHSARecDuoModel')

        if len(hidden_layers) == 0:
            raise ValidationError('MHSARecDuoModel must has at least one value specified for hidden_layers')        
        
        if rec_model_params['rec_model_kwargs'][name]['hidden_size'] == None:
            logger.info('WARNING: RecModel has no value specified for hidden_size. Using n_features value from MHSA')
            rec_model_params['rec_model_kwargs'][name]['hidden_size'] = rec_model_params['mhsa_model_params']['n_features']
        
        self.add_module('rec_model', RecModel(**rec_model_params))

        self.add_module('ffnn', nn.Sequential())
            
        input_dim = rec_model_params['rec_model_kwargs'][name]['hidden_size']
        if rec_model_params['rec_model_kwargs'][name]['bidirectional']:
            input_dim = input_dim * 2
                
        for it, hidden_dim in enumerate(hidden_layers):
            self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
            input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules['mhsa'](x)
        x = self._modules['rec_model'](x)
        x = self._modules['ffnn'](x)
        return x    



class ISSARecDuoModel(nn.Module):

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 issa_model_params: dict,
                 rec_model_params: dict,
                 hidden_layers: list = [],
                 p: float = 0):
        super(ISSARecDuoModel, self).__init__()

        self.add_module('issa', InterlacedSparseSA(**issa_model_params))

        name = list(rec_model_params['rec_model_kwargs'].keys())[0]
        if rec_model_params['rec_model_kwargs'][name]['input_size'] != None:
            raise ValidationError('RecModel must have no value specified for input_size. It is calculated automatically')
        else:
            rec_model_params['rec_model_kwargs'][name]['input_size'] = issa_model_params.n_features

        if len(rec_model_params['hidden_layers']) > 0:
            raise ValidationError('RecModel must have no values specified for hidden_layers. Specify them for ISSARecDuoModel')

        if len(hidden_layers) == 0:
            raise ValidationError('ISSARecDuoModel must has at least one value specified for hidden_layers')        
        
        if rec_model_params['rec_model_kwargs'][name]['hidden_size'] == None:
            logger.info('WARNING: RecModel has no value specified for hidden_size. Using n_features value from ISSA')
            rec_model_params['rec_model_kwargs'][name]['hidden_size'] = rec_model_params['issa_model_params']['n_features']
        
        self.add_module('rec_model', RecModel(**rec_model_params))

        self.add_module('ffnn', nn.Sequential())
            
        input_dim = rec_model_params['rec_model_kwargs'][name]['hidden_size']
        if rec_model_params['rec_model_kwargs'][name]['bidirectional']:
            input_dim = input_dim * 2
                
        for it, hidden_dim in enumerate(hidden_layers):
            self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
            input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules['issa'](x)
        x = self._modules['rec_model'](x)
        x = self._modules['ffnn'](x)
        return x    