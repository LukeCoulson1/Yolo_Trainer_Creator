"""
Configuration management for YOLO Trainer
"""

import json
import os
import yaml
import shutil
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
from datetime import datetime

from . import logger, ensure_directory

class Config:
    """Configuration manager for YOLO Trainer"""

    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config_dir = self.config_file.parent
        self._config = {}
        self._defaults = self._get_defaults()
        ensure_directory(str(self.config_dir))
        self.load()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Application settings
            "app_title": "YOLO Dataset Creator & Trainer v2.1",
            "window_size": "1200x800",

            # Dataset settings
            "datasets_dir": "datasets",
            "default_dataset": "",
            "supported_image_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            "max_image_size": (2048, 2048),
            "validate_on_load": True,

            # Model settings
            "models_dir": "models",
            "default_model": "yolov8n.pt",
            "supported_model_formats": ['.pt', '.pth', '.yaml'],

            # Training settings
            "default_epochs": 50,
            "default_batch_size": 16,
            "default_img_size": 640,
            "default_device": 0,
            "num_workers": 8,
            "cache_images": True,
            "pin_memory": True,

            # UI settings
            "theme": "default",
            "auto_save_annotations": True,
            "show_progress": True,

            # Logging settings
            "log_level": "INFO",
            "log_dir": "logs",
            "max_log_files": 10,

            # Backup settings
            "backup_dir": "backups",
            "auto_backup": True,
            "backup_interval_days": 7,
            "backup_interval": 300,  # seconds

            # Advanced settings
            "enable_gpu_acceleration": True,
            "max_recent_datasets": 10,
            "strict_validation": False,

            # Export settings
            "export_format": "onnx",
            "export_dir": "exports"
        }

    def load(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # Merge with defaults
                self._config = self._defaults.copy()
                self._config.update(loaded_config)

                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                # Use defaults if no config file exists
                self._config = self._defaults.copy()
                logger.info("Using default configuration")
                return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = self._defaults.copy()
            return False

    def save(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            self._config[key] = value

            # Auto-save if enabled
            if self.get('auto_save_annotations', True):
                return self.save()

            return True

        except Exception as e:
            logger.error(f"Failed to set configuration value: {e}")
            return False

    def update(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        try:
            self._config.update(updates)

            # Auto-save if enabled
            if self.get('auto_save_annotations', True):
                return self.save()

            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self._config = self._defaults.copy()
            return self.save()

        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False

    def export_to_yaml(self, yaml_file: Union[str, Path]) -> bool:
        """Export configuration to YAML file"""
        try:
            yaml_file = Path(yaml_file)

            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration exported to {yaml_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration to YAML: {e}")
            return False

    def import_from_yaml(self, yaml_file: Union[str, Path]) -> bool:
        """Import configuration from YAML file"""
        try:
            yaml_file = Path(yaml_file)

            if not yaml_file.exists():
                logger.error(f"YAML file does not exist: {yaml_file}")
                return False

            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                self._config.update(yaml_config)
                return self.save()

            return False

        except Exception as e:
            logger.error(f"Failed to import configuration from YAML: {e}")
            return False

    def create_backup(self) -> Optional[str]:
        """Create a backup of the current configuration"""
        try:
            backup_dir = Path(self.get('backup_dir', 'backups')) / 'config'
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"config_backup_{timestamp}.json"

            shutil.copy2(self.config_file, backup_file)

            logger.info(f"Configuration backup created: {backup_file}")
            return str(backup_file)

        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            return None

    def restore_from_backup(self, backup_file: Union[str, Path]) -> bool:
        """Restore configuration from backup"""
        try:
            backup_file = Path(backup_file)
            if not backup_file.exists():
                logger.error(f"Backup file does not exist: {backup_file}")
                return False

            # Create backup of current config before restore
            current_backup = self.create_backup()

            # Restore from backup
            shutil.copy2(backup_file, self.config_file)

            # Reload configuration
            if self.load():
                logger.info(f"Configuration restored from {backup_file}")
                return True
            else:
                # Restore previous config if load failed
                if current_backup:
                    shutil.copy2(current_backup, self.config_file)
                    self.load()
                return False

        except Exception as e:
            logger.error(f"Failed to restore configuration from backup: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values (with defaults)"""
        result = self._defaults.copy()
        result.update(self._config)
        return result

    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration values"""
        issues = []

        # Validate paths
        path_settings = ['datasets_dir', 'models_dir', 'log_dir', 'backup_dir', 'export_dir']
        for path_setting in path_settings:
            path_value = self.get(path_setting, "")
            if path_value:
                try:
                    Path(path_value)
                except Exception as e:
                    issues.append(f"Invalid path for {path_setting}: {e}")

        # Validate numeric values
        numeric_settings = {
            'default_epochs': (1, 10000),
            'default_batch_size': (1, 512),
            'default_img_size': (32, 4096),
            'default_device': (-1, 16),  # -1 for CPU, 0-16 for GPUs
            'max_log_files': (1, 1000),
            'backup_interval_days': (1, 365),
            'backup_interval': (10, 3600),  # 10 seconds to 1 hour
            'num_workers': (0, 32),
            'max_recent_datasets': (1, 100)
        }

        for setting, (min_val, max_val) in numeric_settings.items():
            value = self.get(setting)
            if value is not None:
                try:
                    num_value = int(value)
                    if not (min_val <= num_value <= max_val):
                        issues.append(f"{setting} must be between {min_val} and {max_val}, got {num_value}")
                except (ValueError, TypeError):
                    issues.append(f"{setting} must be a valid integer")

        # Validate lists
        list_settings = ['supported_image_formats', 'supported_model_formats']
        for setting in list_settings:
            value = self.get(setting)
            if value is not None and not isinstance(value, list):
                issues.append(f"{setting} must be a list")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = self.get('log_level', 'INFO')
        if log_level not in valid_log_levels:
            issues.append(f"log_level must be one of: {', '.join(valid_log_levels)}")

        # Validate theme
        valid_themes = ['default', 'dark', 'light']
        theme = self.get('theme', 'default')
        if theme not in valid_themes:
            issues.append(f"theme must be one of: {', '.join(valid_themes)}")

        return len(issues) == 0, issues

    def validate(self) -> bool:
        """Legacy validate method for backward compatibility"""
        valid, issues = self.validate_config()
        if issues:
            logger.warning("Configuration validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        return valid

# Global configuration instance
config = Config()

# Available YOLO models
AVAILABLE_MODELS = {
    "YOLOv8 Nano": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 XLarge": "yolov8x.pt",
    "YOLOv11 Nano": "yolo11n.pt",
    "YOLOv11 Small": "yolo11s.pt",
    "YOLOv11 Medium": "yolo11m.pt",
    "YOLOv11 Large": "yolo11l.pt",
    "YOLOv11 XLarge": "yolo11x.pt"
}

# Training parameter ranges
TRAINING_RANGES = {
    "epochs": (1, 1000),
    "batch_size": (1, 128),
    "img_size": (64, 2048),
    "learning_rate": (0.0001, 0.1)
}

# UI themes
THEMES = {
    "default": {
        "bg_color": "#f0f0f0",
        "fg_color": "#000000",
        "accent_color": "#0078d4",
        "success_color": "#107c10",
        "warning_color": "#ff8c00",
        "error_color": "#d13438"
    },
    "dark": {
        "bg_color": "#1a1a1a",
        "fg_color": "#ffffff",
        "accent_color": "#0078d4",
        "success_color": "#107c10",
        "warning_color": "#ff8c00",
        "error_color": "#d13438"
    }
}
