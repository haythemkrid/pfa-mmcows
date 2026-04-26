import os
import yaml
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Robustly loads configuration from a YAML or JSON file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format. Use YAML or JSON: {config_path}")
import yaml
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use YAML or JSON.")
