import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.common import get_scalar_log



def smoother(scalars: list[float], weight: float) -> list[float]:
    """
    Function to smooth a list of scalars using exponential moving average method

    Parameters:
    - scalars (list[float]): List of scalars to be smoothed
    - weight (float): Smoothing weight

    Returns:
    - smoothed (list[float]): List of smoothed scalars 
    """
    
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed



def simple_plot(x, 
                y, 
                xlabel: str=None, 
                ylabel: str=None, 
                scatterplot: bool=False, 
                lineplot: bool=True,
                params_dict: dict={'figure.figsize': (20,8), 
                                   'font.size': 20, 
                                   'lines.linewidth': 5, 
                                   'lines.markersize': 15},
                xticks: list=None, 
                yticks: list=None):
    """
    Function to make a single plot for two one-dimensional arrays x and y

    Parameters:
    - x (1d array like): Array to be used along x axis
    - y (1d array like): Array to be used along y axis
    - xlabel (str): Label for x axis
    - ylabel (str): Label for y axis
    - scatterplot (bool): Whether to make a scatterplot
    - lineplot (bool): Whether to make a lineplot
    - params_dict (dict): Dictionary with paramaters for the plot
    - xticks (list): List of ticks for x axis
    - yticks (list): List of ticks for x axis
    """
    
    if params_dict is not None:
        plt.rcParams.update(params_dict)
    plt.figure()
    sns.set_style("darkgrid")
    if lineplot: sns.lineplot(x=x, y=y)
    if scatterplot: sns.scatterplot(x=x, y=y)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    if xticks != None: plt.xticks(xticks)
    if yticks != None: plt.yticks(yticks)    
    plt.show()  



def simple_plot2(X, 
                 Y, 
                 labels, 
                 xlabel=None, 
                 ylabel=None, 
                 scatterplot=False, 
                 lineplot=True,
                 params_dict={'figure.figsize': (20,8), 'font.size': 20, 'lines.linewidth': 5, 'lines.markersize': 15},
                 xlim=None, 
                 ylim=None, 
                 weight=0, 
                 linestyles=None, 
                 colors=None):
    """
    Function to make plots for a set of one-dimensional arrays x and y

    Parameters:
    - X (2d array like): List of arrays to be used along x axis
    - Y (2d array like): Corresponding list of arrays to be used along y axis
    - labels (list): List of corresponding labels 
    - xlabel (str): Label for x axis
    - ylabel (str): Label for y axis
    - scatterplot (bool): Whether to make a scatterplot
    - lineplot (bool): Whether to make a lineplot
    - params_dict (dict): Dictionary with paramaters for the plot
    - xlim (list): Borders of a plot along x axis
    - ylim (list): Borders of a plot along x axis
    - weight (float): Smoothing weight. If 0, no smoothing is applied. If > 0, smothing is applied.
    - linestyles (list): List of corresponding linestyles
    - colors (list): List of corresponding colors
    """
    
    if params_dict is not None:
        plt.rcParams.update(params_dict)
    plt.figure()
    sns.set_style("darkgrid")
    for it, (x, y, label) in enumerate(zip(X, Y, labels)):
        
        linestyle = '-'
        if linestyles != None: linestyle = linestyles[it]
        color = None
        if colors != None: color = colors[it]
            
        y = smoother(y, weight)
        
        if lineplot and not scatterplot: 
            sns.lineplot(x=x, y=y, label=label, linestyle=linestyle, color=color)
        if not lineplot and scatterplot: 
            sns.scatterplot(x=x, y=y, label=label, color=color)
        if lineplot and scatterplot: 
            sns.lineplot(x=x, y=y, label=label, linestyle=linestyle, color=color)
            sns.scatterplot(x=x, y=y, color=color)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    if xlim != None: plt.xlim(*xlim)
    if ylim != None: plt.ylim(*ylim)
    plt.legend()
    plt.show()    



def plot_tf_logs(models: list, 
                 weight: float=0,
                 dataset: str ='validation', 
                 linestyles='-',
                 colors: list=['black', 'purple', 'orange', 'red', 'green', 'blue', 'grey'],
                 vars_to_plot: dict={'CrossEntropyLoss': [0.41, 0.6],
                                     'accuracy': [0.7, 0.9],
                                     'f1_neg': [0.7, 0.82]},
                 threshold: float=0,
                 digits: int=6):
    """
    Function to make plots of tf scalar logs for specified models.
    Also prints mean logs.

    Parameters:
    - models: List of models.
    - weight (float): Smoothing weight. If 0, no smoothing is applied. If > 0, smothing is applied.
    - dataset (str): If 'validation', validation logs are showed. If 'train', training logs are showed.
    - linestyles (list or str): List of corresponding linestyles. If str, then the same linestyle is used.
    - colors (list): List of corresponding colors.s
    - vars_to_plot (dict): Dictionary with scalars to plot with corresponding borders.
    - threshold (float): Rate of data to be skipped when calculating mean logs 
                         (e.g. if 0.5, then first 50% of logs are skipped)
    - digits (int): Number of digits to keep when rounding.
    """
    
    logs = [get_scalar_log(f'logs/callback_head_model_{model}') for model in models]

    if type(linestyles) == str:
        linestyles = [linestyles for _ in range(len(models))]
    
    for name in vars_to_plot.keys():
        y = f'{name}_{dataset}'
        ylabel = '_'.join(y.split('_')[:-1])
        if ('neg' in ylabel) or ('pos' in ylabel):
            ylabel = ylabel.split('_')[0]
        simple_plot2(X=[log[y]['step'] for log in logs], 
                     Y=[log[y]['value'] for log in logs], 
                     labels=[f'{dataset}, {model}' if model != 'Linear' else f'{dataset}, FFNN' for model in models], 
                     xlabel='step', 
                     ylabel=ylabel, 
                     ylim=vars_to_plot[name], 
                     weight=weight,
                     linestyles=linestyles,
                     colors=colors)

    for name in vars_to_plot.keys():
        print('-'*50)
        print(f'Mean values for {name}_{dataset}')
        for (model, log) in zip(models, logs):
            threshold
            current_log = log[f"{name}_{dataset}"]["value"]
            current_log = current_log[int(threshold*len(current_log)):]
            print(f'- {model}: {round(np.mean(current_log), digits)}')