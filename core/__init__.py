"""
YOLO Dataset Creator & Trainer
A comprehensive GUI application for creating YOLO datasets, annotating images,
training models, and testing them - all in one place.

Version: 2.1 (Modular Architecture)
Author: GitHub Copilot
Date: August 28, 2025

Features:
- Dataset creation and management
- Image annotation with bounding boxes
- YOLO model training with RTX 5090 support
- Model fine-tuning and iterative training
- Custom model naming
- Comprehensive results visualization
- Real-time model testing
- Label validation and auto-fixing
"""

__version__ = "2.1.0"
__author__ = "GitHub Copilot"
__date__ = "2025-08-28"

# Standard library imports
import os
import sys
import json
import glob
import shutil
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Third-party imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2

# YOLO imports (with error handling)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOTrainerError(Exception):
    """Custom exception for YOLO Trainer specific errors"""
    pass

class ConfigurationError(YOLOTrainerError):
    """Configuration related errors"""
    pass

class DatasetError(YOLOTrainerError):
    """Dataset related errors"""
    pass

class TrainingError(YOLOTrainerError):
    """Training related errors"""
    pass

def validate_environment():
    """Validate that the environment is properly set up"""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")

    # Check required packages
    required_packages = ['PIL', 'tkinter', 'numpy', 'cv2']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")

    # Check YOLO availability
    if not YOLO_AVAILABLE:
        issues.append("ultralytics package not available")

    # Check directories
    required_dirs = ['datasets', 'models']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                logger.info(f"Created directory: {dir_name}")
            except Exception as e:
                issues.append(f"Cannot create directory {dir_name}: {e}")

    if issues:
        error_msg = "Environment validation failed:\n" + "\n".join(f"• {issue}" for issue in issues)
        logger.error(error_msg)
        raise ConfigurationError(error_msg)

    logger.info("Environment validation passed")
    return True

def create_backup(filename: str, backup_dir: str = "backups") -> str:
    """Create a backup of a file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{Path(filename).stem}_backup_{timestamp}{Path(filename).suffix}"
    backup_path = os.path.join(backup_dir, backup_name)

    shutil.copy2(filename, backup_path)
    logger.info(f"Backup created: {backup_path}")
    return backup_path

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be filesystem-safe"""
    import re

    # Remove/replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')

    # Ensure not empty
    if not filename:
        return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Limit length
    if len(filename) > 100:
        filename = filename[:100]

    return filename

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime to consistent string"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Cannot create directory {path}: {e}")
        return False

# Validate environment on import
try:
    validate_environment()
except ConfigurationError as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)

# Import core modules
try:
    from .config import config, AVAILABLE_MODELS, TRAINING_RANGES, THEMES
    from .dataset_manager import dataset_manager
    from .training_manager import training_manager
    from .ui_components import *
    from .validation import *
    from .error_handler import *
    from .model_comparison import model_comparator
    from .data_augmentation import data_augmentor
    from .experiment_tracking import experiment_tracker

    # Make key components available at package level
    __all__ = [
        # Core functionality
        'logger',
        'config',
        'dataset_manager',
        'training_manager',
        'model_comparator',
        'data_augmentor',
        'experiment_tracker',

        # Utility functions
        'sanitize_filename',
        'format_timestamp',
        'ensure_directory',
        'create_backup',
        'get_file_size_mb',

        # UI components
        'ProgressDialog',
        'ConfirmationDialog',
        'FileSelector',
        'LogViewer',
        'StatusBar',
        'TrainingMonitor',
        'center_window',
        'show_error_message',
        'show_info_message',
        'show_warning_message',
        'ask_yes_no',

        # Validation
        'DataValidator',
        'TrainingValidator',
        'FileValidator',
        'validate_all_requirements',

        # Error handling
        'error_handler',
        'recovery_manager',
        'safe_execute',
        'safe_execute_with_fallback',
        'log_operation_start',
        'log_operation_end',
        'OperationContext',

        # Constants
        'AVAILABLE_MODELS',
        'TRAINING_RANGES',
        'THEMES',
        'DEFAULT_CONFIG',

        # Version info
        '__version__',
        '__author__',
        '__date__'
    ]

except ImportError as e:
    logger.warning(f"Some core modules not available: {e}")
    # Provide basic functionality if modules fail to import
    __all__ = ['logger', 'DEFAULT_CONFIG', '__version__', '__author__', '__date__']
