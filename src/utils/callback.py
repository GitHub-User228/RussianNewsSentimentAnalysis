import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from sklearn.metrics import get_scorer
from tqdm.auto import tqdm

from src.entity.dataset import CustomDataset



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class Callback:
    
    def __init__(self, 
                 writer: SummaryWriter, 
                 dataset: CustomDataset,
                 loss_function: object, 
                 loss_function_name: str,
                 metrics_names: list,
                 eval_steps: int, 
                 batch_size: int):
        
        self.step = 0
        self.writer = writer
        self.eval_steps = eval_steps
        self.loss_function = loss_function
        self.loss_function_name = loss_function_name
        self.metrics_names = metrics_names
        self.metrics = dict(zip(metrics_names, [get_scorer(metric_name) for metric_name in metrics_names]))
        self.batch_size = batch_size
        self.dataset = dataset

    
    def forward(self, model, loss, pred, real):
        
        self.step += 1
        self.writer.add_scalars(self.loss_function_name, {'train': loss}, self.step)

        for (metric_name, metric_func) in self.metrics.items():
            if metric_name in ['precision', 'recall', 'f1']:
                train_score = metric_func._score_func(y_true=real, y_pred=pred)
                self.writer.add_scalars(metric_name+'_pos', {'train': train_score}, self.step)  
                train_score = metric_func._score_func(y_true=1 - real, y_pred=1 - pred)
                self.writer.add_scalars(metric_name+'_neg', {'train': train_score}, self.step)                 
            else:
                train_score = metric_func._score_func(y_true=real, y_pred=pred)
                self.writer.add_scalars(metric_name, {'train': train_score}, self.step)

        if self.eval_steps != None:
            
            if self.step == 1:
                self.writer.add_graph(model, self.dataset[0:1][0].to(model.device))
        
        
            if (self.step % self.eval_steps == 0) or (self.step==1):          
                
                batch_generator = torch.utils.data.DataLoader(dataset=self.dataset, 
                                                              batch_size=self.batch_size,
                                                              shuffle=True)
                iterations = tqdm(enumerate(batch_generator), 
                                  desc='Evaluation', 
                                  leave=False, 
                                  total=math.ceil(batch_generator.dataset.__len__()/batch_generator.batch_size))         
                pred = []
                real = []
                validation_loss = 0
                for it, (batch_X, batch_y) in iterations:
                    with torch.no_grad():
                        batch_X = batch_X.to(model.device)
                        batch_y = batch_y.to(model.device)
    
                        logits = model(batch_X)
    
                        validation_loss += self.loss_function(logits, batch_y.to(torch.int64)).cpu().item()*len(batch_y)
    
                        pred += logits.argmax(axis=1).cpu().tolist()
                        real += batch_y.cpu().tolist()
                
                validation_loss /= len(self.dataset)
                
                
                self.writer.add_scalars(self.loss_function_name, {'validation': validation_loss}, self.step)
                
                for (metric_name, metric_func) in self.metrics.items():
                    if metric_name in ['precision', 'recall', 'f1']:
                        validation_score = metric_func._score_func(y_true=real, y_pred=pred)
                        self.writer.add_scalars(metric_name+'_pos', {'validation': validation_score}, self.step)  
                        validation_score = metric_func._score_func(y_true=1-np.array(real), y_pred=1-np.array(pred))
                        self.writer.add_scalars(metric_name+'_neg', {'validation': validation_score}, self.step)   
                    else:
                        validation_score = metric_func._score_func(y_true=real, y_pred=pred)
                        self.writer.add_scalars(metric_name, {'validation': validation_score}, self.step)
    
                self.writer.add_text('REPORT/validation', str(classification_report(real, pred, digits=6)), self.step)
        
            
    def __call__(self, model, loss, pred, true):
        
        return self.forward(model, loss, pred, true)