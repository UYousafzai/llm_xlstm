"""
_summary_
"""

import torch
import torch.nn as nn

from src.utils import read_yaml_file, load_all_config, init_weights
from src.dataloader import TestingDatasetWikiGerman
from pipelines import Pipeline, GermanWikiPipeline
from src.xLSTM import xLSTM_Sample_Architecture

class Experiment:
    def __init__(self, experiment_config = None, pipeline_type: Pipeline = None):
        self.experiment_config = read_yaml_file(experiment_config)
        self.pipeline_type = pipeline_type

    def warmup(self):
        pass

    def run(self):
        pass



class GermanWikiTestExperiment(Experiment):
    def __init__(self, experiment_config = "./configs/experiment/v1.yaml", pipeline_type = GermanWikiPipeline):
        super().__init__(experiment_config=experiment_config, pipeline_type=pipeline_type)
    

    def warmup(self):
        self.g_config = load_all_config(self.experiment_config["experiment"]["base_config"], self.experiment_config["experiment"]["model_config"], self.experiment_config["experiment"]["training_config"])
        self.dataset = TestingDatasetWikiGerman(self.experiment_config["experiment"]["txt_file_path"])
        self.criterion = nn.CrossEntropyLoss()
        self.model = xLSTM_Sample_Architecture(len(self.dataset.vocab), self.g_config.embedding_size, self.g_config.hidden_size, 
              self.g_config.num_layers, self.g_config.num_blocks).to(self.g_config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.g_config.learning_rate)
        self.model.apply(init_weights)
        self.pipeline = self.pipeline_type(config = self.g_config, dataset = self.dataset, model = self.model, optimizer = self.optimizer, criterion = self.criterion)


    def run(self):
        self.pipeline.train()






