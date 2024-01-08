import torch
from torch import nn
from src.models.linear import LinearBlock



class RecModel(nn.Module):

    RECURRENT_MODELS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 rec_model_kwargs: dict, 
                 hidden_layers: list = [], 
                 p: float = 0.):
        super(RecModel, self).__init__()

        self.rec_model_name = list(rec_model_kwargs.keys())[0]
        self.bidirectional = rec_model_kwargs[self.rec_model_name]['bidirectional']
        self.add_module(self.rec_model_name, self.RECURRENT_MODELS[self.rec_model_name](**rec_model_kwargs[self.rec_model_name]))

        if len(hidden_layers) > 0:
            self.add_module('ffnn', nn.Sequential())
            
            input_dim = rec_model_kwargs[self.rec_model_name]['hidden_size']
            if self.bidirectional:
                input_dim = input_dim * 2
                
            for it, hidden_dim in enumerate(hidden_layers):
                self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
                input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules[self.rec_model_name](x)[0][:, -1, :]
        if 'ffnn' in self._modules:
            x = self._modules['ffnn'](x)
        return x 



class ParallelRecModels(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 rec_models_kwargs: dict,
                 hidden_layers: list = [], 
                 p: float = 0.):
        super(ParallelRecModels, self).__init__()

        self.rec_models_names = list(rec_models_kwargs.keys())
        self.bidirectional = dict([(k, v['rec_model_kwargs'][k]['bidirectional']) for (k, v) in rec_models_kwargs.items()])
        self.add_module('recurrent_models', nn.ModuleDict(dict([(k, RecModel(**v)) for (k, v) in rec_models_kwargs.items()])))

        if len(hidden_layers) > 0:
            
            input_dim = 0
            for (k, v) in rec_models_kwargs.items():
                if 'hidden_layers' in v.keys():
                    if len(v['hidden_layers']) > 0:
                        input_dim += v['hidden_layers'][-1]
                else:
                    input_dim = v['rec_model_kwargs'][k]['hidden_size']*(1 + self.bidirectional[k])
    
            self.add_module('ffnn', nn.Sequential())       
            for it, hidden_dim in enumerate(hidden_layers):
                self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
                input_dim = hidden_dim

    
    def forward(self, x):
        x = torch.cat([self._modules['recurrent_models'][k](x) for k in self.rec_models_names], axis=-1)
        if 'ffnn' in self._modules:
            x = self._modules['ffnn'](x)
        return x 