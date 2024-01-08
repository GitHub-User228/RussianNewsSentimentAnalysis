"""
common.py

Purpose:
    Contains common functionalities used across the project.
"""

from pathlib import Path
from typing import Any, List
import os
import sys
import yaml
import json
import joblib

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from typing import Union

import gc
import torch
from tensorboard.backend.event_processing import event_accumulator

from src import logger



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file, and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the yaml file.

    Raises:
        ValueError: If the yaml file is empty.
        e: If any other exception occurs.

    Returns:
        ConfigBox: The yaml content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.info("Value exception: empty yaml file")
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.info(f"An exception {e} has occurred")
        raise e



@ensure_annotations
def save_yaml(path: Path, data: dict):
    """
    Save yaml data

    Args:
        path (Path): path to yaml file
        data (dict): data to be saved in yaml file
    """
    try:
        with open(path, "w") as f:
            yaml.dump(data, f, indent=4)
        logger.info(f"yaml file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save yaml to {path}. Error: {e}")
        raise
        


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save json to {path}. Error: {e}")
        raise



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to read {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to load json from {path}. Error: {e}")
        raise



@ensure_annotations
def save_tensor(path: str, filename: str, data: torch.Tensor, logging: bool=True):
    """
    Save json data

    Args:
        path (str): path
        filename (str): filename
        data (torch.Tensor): tensor to be saved
    """
    try:
        torch.save(data, os.path.join(path, filename))
        if logging:
            logger.info(f"tensor file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save tensor to {path}. Error: {e}")
        raise



@ensure_annotations
def load_tensor(path: str, filename: str, logging: bool=True) -> torch.Tensor:
    """
    Save json data

    Args:
        path (str): path
        filename (str): filename

    Returns:
        data (torch.Tensor): Loaded tensor
    """
    try:
        data = torch.load(os.path.join(path, filename))
        if logging:
            logger.info(f"tensor file has been loaded from: {path}")
        return data
    except PermissionError:
        logger.error(f"Permission denied to read from {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to read tensor from {path}. Error: {e}")
        raise



def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()



def get_scalar_log(path: str) -> dict:
    """
    Loads all scalar logs (from tensorflow notebooks) from specified directory

    Parameters:
    - path (str): Path to directory with logs.

    Returns:
    - output (dict): Dictionary with all logs.
    """
    dirs = [k for k in os.listdir(path) if '.' not in k]
    output = {}
    for dir in dirs:
        filename = os.listdir(os.path.join(path, dir))[0]
        ea = event_accumulator.EventAccumulator(os.path.join(os.path.join(path, dir), filename))
        ea.Reload()
        tag = ea.Tags()['scalars'][0]
        logs = ea.Scalars(tag)
        output[dir] = {}
        output[dir]['step'] = [log.step for log in logs]
        output[dir]['value'] = [log.value for log in logs]
    return output



def save_pkl(model, path: str, filename: str):
    """
    Saves second level model (from sklearn) as pkl file

    Parameters:
    - model: Second level model to be saved
    - path (str): Path where to save model
    - filename (str): Filename of pkl file.
    """
    
    with open(os.path.join(path, filename), 'wb') as f:
        joblib.dump(model, f)



def load_pkl(path: str, filename: str):
    """
    Loads second level model

    Parameters:
    - path (str): Path from where to load model
    - filename (str): Filename of pkl file.

    Returns:
    - model: Loaded second level model
    """
    
    return joblib.load(os.path.join(path, filename))