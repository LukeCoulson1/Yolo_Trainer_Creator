"""
Validation utilities for YOLO Trainer
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import re

from . import logger

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_yaml_file(file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate YAML file structure"""
        issues = []

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return False, [f"File does not exist: {file_path}"]

            if not file_path.is_file():
                return False, [f"Path is not a file: {file_path}"]

            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if data is None:
                return False, ["YAML file is empty or invalid"]

            # Validate required fields for YOLO data.yaml
            required_fields = ['nc', 'names']
            for field in required_fields:
                if field not in data:
                    issues.append(f"Missing required field: {field}")

            if 'nc' in data and not isinstance(data['nc'], int):
                issues.append("Field 'nc' must be an integer")

            if 'names' in data and not isinstance(data['names'], list):
                issues.append("Field 'names' must be a list")

            if 'names' in data and isinstance(data['names'], list):
                if len(data['names']) != data.get('nc', 0):
                    issues.append(f"Number of names ({len(data['names'])}) doesn't match nc ({data.get('nc', 0)})")

            return len(issues) == 0, issues

        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"]
        except Exception as e:
            return False, [f"Error validating YAML: {e}"]

    @staticmethod
    def validate_dataset_directory(dataset_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate dataset directory structure"""
        issues = []
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            return False, [f"Dataset directory does not exist: {dataset_path}"]

        if not dataset_path.is_dir():
            return False, [f"Path is not a directory: {dataset_path}"]

        # Check for required subdirectories
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(f"Missing required directory: {dir_name}")
            elif not dir_path.is_dir():
                issues.append(f"Path is not a directory: {dir_name}")

        # Check for data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            issues.append("Missing data.yaml file")
        else:
            yaml_valid, yaml_issues = DataValidator.validate_yaml_file(data_yaml)
            if not yaml_valid:
                issues.extend([f"data.yaml: {issue}" for issue in yaml_issues])

        # Check for image/label consistency
        if (dataset_path / "images").exists() and (dataset_path / "labels").exists():
            image_files = set()
            label_files = set()

            # Collect image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                for img_file in (dataset_path / "images").glob(ext):
                    image_files.add(img_file.stem)

            # Collect label files
            for lbl_file in (dataset_path / "labels").glob("*.txt"):
                label_files.add(lbl_file.stem)

            orphaned_labels = label_files - image_files
            missing_labels = image_files - label_files

            if orphaned_labels:
                issues.append(f"Orphaned label files (no corresponding images): {len(orphaned_labels)}")

            if missing_labels:
                issues.append(f"Images without labels: {len(missing_labels)}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_label_file(label_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate YOLO label file format"""
        issues = []

        try:
            label_path = Path(label_path)
            if not label_path.exists():
                return False, [f"Label file does not exist: {label_path}"]

            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"Line {i}: Expected 5 values, got {len(parts)}")
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate ranges
                    if class_id < 0:
                        issues.append(f"Line {i}: Class ID must be non-negative")

                    for j, (name, value) in enumerate([('x_center', x_center), ('y_center', y_center),
                                                     ('width', width), ('height', height)], 1):
                        if not (0.0 <= value <= 1.0):
                            issues.append(f"Line {i}: {name} must be between 0.0 and 1.0, got {value}")

                except ValueError as e:
                    issues.append(f"Line {i}: Invalid numeric value - {e}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Error validating label file: {e}"]

    @staticmethod
    def validate_image_file(image_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate image file"""
        issues = []

        try:
            from PIL import Image

            image_path = Path(image_path)
            if not image_path.exists():
                return False, [f"Image file does not exist: {image_path}"]

            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            if image_path.suffix.lower() not in valid_extensions:
                issues.append(f"Unsupported image format: {image_path.suffix}")

            # Try to open image
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                issues.append(f"Corrupted or invalid image file: {e}")

            return len(issues) == 0, issues

        except ImportError:
            # PIL not available, just check file existence and extension
            image_path = Path(image_path)
            if not image_path.exists():
                return False, [f"Image file does not exist: {image_path}"]

            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            if image_path.suffix.lower() not in valid_extensions:
                issues.append(f"Unsupported image format: {image_path.suffix}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Error validating image file: {e}"]

    @staticmethod
    def validate_model_path(model_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate YOLO model path"""
        issues = []

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False, [f"Model file does not exist: {model_path}"]

            if not model_path.is_file():
                return False, [f"Path is not a file: {model_path}"]

            # Check file extension
            valid_extensions = {'.pt', '.pth', '.yaml', '.yml'}
            if model_path.suffix.lower() not in valid_extensions:
                issues.append(f"Unsupported model format: {model_path.suffix}")

            # Check file size (basic sanity check)
            file_size = model_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                issues.append("Model file seems too small to be valid")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Error validating model path: {e}"]

class TrainingValidator:
    """Training configuration validation"""

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training configuration"""
        issues = []

        # Required fields
        required_fields = ['data_yaml', 'model_path']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
            elif not config[field]:
                issues.append(f"Required field '{field}' is empty")

        # Validate data.yaml
        if 'data_yaml' in config and config['data_yaml']:
            yaml_valid, yaml_issues = DataValidator.validate_yaml_file(config['data_yaml'])
            if not yaml_valid:
                issues.extend([f"data.yaml: {issue}" for issue in yaml_issues])

        # Validate model path
        if 'model_path' in config and config['model_path']:
            model_valid, model_issues = DataValidator.validate_model_path(config['model_path'])
            if not model_valid:
                issues.extend([f"model_path: {issue}" for issue in model_issues])

        # Validate numeric fields
        numeric_fields = {
            'epochs': (1, 10000),
            'batch_size': (1, 512),
            'img_size': (32, 4096),
            'save_period': (1, 1000),
            'workers': (0, 32),
            'patience': (1, 1000)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                try:
                    value = int(config[field])
                    if not (min_val <= value <= max_val):
                        issues.append(f"Field '{field}' must be between {min_val} and {max_val}, got {value}")
                except (ValueError, TypeError):
                    issues.append(f"Field '{field}' must be a valid integer")

        # Validate learning rates
        lr_fields = ['lr0', 'lrf']
        for field in lr_fields:
            if field in config:
                try:
                    value = float(config[field])
                    if not (0.0 < value <= 1.0):
                        issues.append(f"Field '{field}' must be between 0.0 and 1.0, got {value}")
                except (ValueError, TypeError):
                    issues.append(f"Field '{field}' must be a valid number")

        return len(issues) == 0, issues

class FileValidator:
    """File and path validation utilities"""

    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, List[str]]:
        """Validate filename for safety"""
        issues = []

        if not filename:
            return False, ["Filename cannot be empty"]

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in filename:
                issues.append(f"Filename contains invalid character: '{char}'")

        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in reserved_names:
            issues.append(f"Filename '{name_without_ext}' is reserved by the system")

        # Check length
        if len(filename) > 255:
            issues.append("Filename is too long (max 255 characters)")

        return len(issues) == 0, issues

    @staticmethod
    def validate_directory_path(path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate directory path"""
        issues = []

        try:
            path = Path(path)

            # Check if path exists
            if not path.exists():
                issues.append(f"Path does not exist: {path}")
                return False, issues

            # Check if it's a directory
            if not path.is_dir():
                issues.append(f"Path is not a directory: {path}")
                return False, issues

            # Check permissions
            try:
                # Try to list contents
                list(path.iterdir())
            except PermissionError:
                issues.append(f"Permission denied: {path}")
            except Exception as e:
                issues.append(f"Error accessing directory: {e}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Error validating directory path: {e}"]

def validate_all_requirements() -> Tuple[bool, Dict[str, List[str]]]:
    """Validate all system requirements"""
    issues = {}

    # Check Python packages
    required_packages = ['ultralytics', 'torch', 'PIL', 'opencv-python', 'pandas', 'pyyaml']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        issues['packages'] = [f"Missing required packages: {', '.join(missing_packages)}"]

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            issues['cuda'] = [f"CUDA available: {torch.cuda.get_device_name(0)}"]
        else:
            issues['cuda'] = ["CUDA not available - training will be slower on CPU"]
    except Exception as e:
        issues['cuda'] = [f"Error checking CUDA: {e}"]

    return len([k for k, v in issues.items() if k != 'cuda' or 'not available' in str(v)]) == 0, issues
