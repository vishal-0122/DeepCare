import os
import yaml

class Config:
    """Utility class to load and access project configuration from YAML"""

    def __init__(self, config_path: str = None):
        # default path if not provided
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "configs", "config.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file {self.config_path}: {e}")

    def get(self, key: str, default=None, strict=False):
        """Get a config value using dot notation (e.g. 'data.patient_demographics.path')"""
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            if strict:
                raise KeyError(f"Config key '{key}' not found in {self.config_path}")
            return default

    def get_dict(self):
        """Return the full config as a dictionary"""
        return self.config


# single instance so we don't reload the config multiple times
config = Config()
