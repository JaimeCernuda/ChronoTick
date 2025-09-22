"""
Configuration management for TSFM library.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for TSFM library."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary. If None, loads default config.
        """
        if config_dict is None:
            config_dict = self._get_default_config()
        
        self._config = config_dict
        self._validate_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "models": {
                "timesfm": {
                    "model_repo": "google/timesfm-1.0-200m",
                    "context_len": 512,
                    "horizon_len": 96,
                    "input_patch_len": 32,
                    "output_patch_len": 128,
                    "num_layers": 20,
                    "model_dims": 1280
                },
                "ttm": {
                    "model_repo": "ibm-granite/granite-timeseries-ttm-v1",
                    "context_length": 512,
                    "prediction_length": 96
                },
                "chronos": {
                    "model_size": "tiny",
                    "model_repo": "amazon/chronos-t5-tiny",
                    "prediction_length": 96,
                    "num_samples": 20
                },
                "toto": {
                    "model_repo": "Datadog/Toto-Open-Base-1.0",
                    "prediction_length": 336,
                    "num_samples": 256,
                    "samples_per_batch": 256
                },
                "time_moe": {
                    "variant": "50M",
                    "model_repo": "Maple728/TimeMoE-50M",
                    "max_context_length": 4096,
                    "prediction_length": 96
                }
            },
            "datasets": {
                "default_normalization": "standard",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "testing": {
                "default_context_length": 512,
                "default_horizon": 96,
                "benchmark_datasets": ["ETTh1", "ETTh2"],
                "metrics": ["mae", "rmse", "mape"],
                "tolerance": {
                    "mae": 1000.0,
                    "rmse": 1000.0,
                    "mape": 1000.0
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _validate_config(self) -> None:
        """Validate configuration structure."""
        required_sections = ["models", "datasets", "testing", "logging"]
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing config section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.timesfm.context_len')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def has(self, key: str) -> bool:
        """
        Check if configuration has a key.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if not key or key.strip() == '':
            raise ValueError("Configuration key cannot be empty")
        
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for a specific model."""
        return self.get(f"models.{model_name}", {})
    
    def update(self, other_config: Dict) -> None:
        """Update configuration with another dictionary."""
        self._update_recursive(self._config, other_config)
    
    def _update_recursive(self, base: Dict, update: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_recursive(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def __repr__(self):
        return f"Config({list(self._config.keys())})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return Config(config_dict)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Config object to save
        config_path: Path to save configuration file
    """
    try:
        # Create directory if it doesn't exist
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                json.dump(config.to_dict(), f, indent=2)
            else:
                # Default to YAML
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        _merge_recursive(result, config)
    
    return result


def _merge_recursive(base: Dict, update: Dict) -> None:
    """Recursively merge dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_recursive(base[key], value)
        else:
            base[key] = value


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    
    # Validate logging level if present
    if 'logging' in config and 'level' in config['logging']:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config['logging']['level'] not in valid_levels:
            raise ValueError(f"Invalid logging level: {config['logging']['level']}")
    
    # Validate device settings if present
    if 'models' in config:
        for model_name, model_config in config['models'].items():
            if isinstance(model_config, dict) and 'device' in model_config:
                valid_devices = ['cpu', 'cuda', 'auto', 'mps']
                if model_config['device'] not in valid_devices:
                    raise ValueError(f"Invalid device for {model_name}: {model_config['device']}")
    
    return True


def get_default_config() -> Dict:
    """
    Get default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    config = Config()
    return config.to_dict()