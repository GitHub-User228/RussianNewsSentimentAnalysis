import torch
from torch import int64
from torch import tensor
from torch.utils.data import Dataset



class CustomDataset(Dataset):

    
    def __init__(self, input_ids, attention_mask, y):

        super(CustomDataset, self).__init__()

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dtype == torch.int32:
               self.input_ids = input_ids
            else:
               self.input_ids = input_ids.to(torch.int32) 
        else:
            self.input_ids = tensor(input_ids).to(torch.int32)

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dtype == torch.int8:
               self.attention_mask = attention_mask
            else:
               self.attention_mask = attention_mask.to(torch.int8) 
        else:
            self.attention_mask = tensor(attention_mask).to(torch.int8)

        if isinstance(y, torch.Tensor):
            if y.dtype == torch.int8:
               self.y = y
            else:
               self.y = y.to(torch.int8) 
        else:
            self.y = tensor(y).to(torch.int8)

    
    def __len__(self):
        return len(self.y)

    
    def __getitem__(self, i):
        return self.input_ids[i,:], self.attention_mask[i,:], self.y[i]



class CustomDataset2(Dataset):

    
    def __init__(self, input_ids, attention_mask):

        super(CustomDataset2, self).__init__()
        
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dtype == torch.int32:
               self.input_ids = input_ids
            else:
               self.input_ids = input_ids.to(torch.int32) 
        else:
            self.input_ids = tensor(input_ids).to(torch.int32)

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dtype == torch.int8:
               self.attention_mask = attention_mask
            else:
               self.attention_mask = attention_mask.to(torch.int8) 
        else:
            self.attention_mask = tensor(attention_mask).to(torch.int8)
            

    def __len__(self):
        return self.input_ids.shape[0]

    
    def __getitem__(self, i):
        return self.input_ids[i,:], self.attention_mask[i,:]



class CustomDataset3(Dataset):

    
    def __init__(self, X, y):

        super(CustomDataset3, self).__init__()

        if isinstance(X, torch.Tensor):
            if X.dtype == torch.float32:
               self.X = X
            else:
               self.X = X.to(torch.float32) 
        else:
            self.X = tensor(X).to(torch.float32)

        if isinstance(y, torch.Tensor):
            if y.dtype == torch.int8:
               self.y = y
            else:
               self.y = y.to(torch.int8) 
        else:
            self.y = tensor(y).to(torch.int8)

    
    def __len__(self):
        return len(self.y)

    
    def __getitem__(self, i):
        return self.X[i,:], self.y[i]



class CustomDataset4(Dataset):

    
    def __init__(self, X):

        super(CustomDataset4, self).__init__()

        if isinstance(X, torch.Tensor):
            if X.dtype == torch.float32:
               self.X = X
            else:
               self.X = X.to(torch.float32) 
        else:
            self.X = tensor(X).to(torch.float32)
    
    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, i):
        return self.X[i,:]