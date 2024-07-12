import yaml
import torch
from typing import Dict, Any

class Config:
    def __init__(self, base_config: str, model_config: str, training_config: str):
        self.config = self._load_configs(base_config, model_config, training_config)
        self._process_config()

    def _load_configs(self, *config_files: str) -> Dict[str, Any]:
        config = {}
        for file in config_files:
            with open(file, 'r') as f:
                config.update(yaml.safe_load(f))
        return config

    def _process_config(self):
        if self.config['device'] == 'auto':
            self.config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    def __getattr__(self, key: str) -> Any:
        return self.config.get(key)

    def __getitem__(self, key: str) -> Any:
        return self.config.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.config

def load_all_config(base_config: str, model_config: str, training_config: str) -> Config:
    return Config(base_config, model_config, training_config)