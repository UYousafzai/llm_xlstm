"""
General Utilities
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import yaml

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the available device (CPU, CUDA, or MPS) for running the model.
    
    Returns:
        torch.device: The available device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_checkpoint(model, optimizer, checkpoint_dir, epoch):
    """
    Save the model and optimizer state as a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        checkpoint_dir (str): The directory to save the checkpoint.
        epoch (int): The current epoch number.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def init_weights(m):
    if type(m) in [nn.Linear, nn.Embedding]:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)




def load_checkpoint(model, optimizer, checkpoint_path, device=None):
    """
    Load the model and optimizer state from a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str): The path to the checkpoint file.
        device (torch.device, optional): The device to load the model onto.
    
    Returns:
        tuple: A tuple containing the loaded model, optimizer, and the epoch number.
    """
    if device is None:
        device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return model, optimizer, epoch