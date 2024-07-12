"""_summary_

Returns:
    _type_: _description_
"""
# include the parent directory for imports
import os
import sys

import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# torch imports for training
import torch
import torch.nn as nn

# pipeline required imports
from src.xLSTM import XLSTM_Model
from src.utils import check_nan, Config
from src.dataloader import BaseDataset

class Pipeline:
    def __init__(self, config: Config = None, dataset: BaseDataset = None, model = None, optimizer = None, criterion = None):
        self.config = config
        self.dataset = dataset
        self.model = model

    def warmup(self):
        pass

    def train(self):
        pass


class GermanWikiPipeline(Pipeline):
    def __init__(self, model: XLSTM_Model = None, optimizer: torch.optim.Optimizer = None, criterion: nn.modules.loss._Loss = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def warmup(self):
        pass

    def train(self):
        start_time = time.time()
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
        
            for batch_idx, batch in enumerate(self.dataset.train_loader):
            
                self.optimizer.zero_grad()
                input_seq = batch[:, :-1].to(self.config['device'])
                target_seq = batch[:, 1:].to(self.config['device'])

                print("this was the call1")
                if check_nan(input_seq, "input_seq"):
                    break
                
                output, _ = self.model(input_seq)
                
                print("this was the call12")
                if check_nan(output, "model output"):
                    break
                
                output = output.contiguous().view(-1, len(self.dataset.vocab))
                target_seq = target_seq.contiguous().view(-1)
                
                loss = self.criterion(output, target_seq)
                print("Batch loss was for batch number {batch_idx}: ", loss)

                
                print("this was the call13")
                if check_nan(loss, "loss"):
                    break
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_value'])
                
                # Check gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if check_nan(param.grad, f"gradient of {name}"):
                            break
                
                self.optimizer.step()
                
                # Check parameters after update
                for name, param in self.model.named_parameters():
                    if check_nan(param, f"parameter {name} after update"):
                        break
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Batch {batch_idx+1}/{len(self.dataset.train_loader)}, Loss: {loss.item():.4f}")
            
            if math.isnan(total_loss):
                print("NaN detected in total_loss. Stopping training.")
                break
            
            avg_loss = total_loss / len(self.dataset.train_loader)
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Average Loss: {avg_loss:.4f}")

        end_time = time.time()
        print(f"Training completed! Total time: {end_time - start_time:.2f} seconds")
