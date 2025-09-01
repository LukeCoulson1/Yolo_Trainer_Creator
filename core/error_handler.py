"""
Error handling and recovery utilities for YOLO Trainer
"""

import traceback
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

from . import logger, format_timestamp, ensure_directory

class ErrorHandler:
    """Centralized error handling and recovery"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.error_log_file = self.log_dir / "errors.log"
        self.recovery_actions: Dict[str, Callable] = {}
        ensure_directory(str(self.log_dir))

    def register_recovery_action(self, error_type: str, action: Callable):
        """Register a recovery action for a specific error type"""
        self.recovery_actions[error_type] = action

    def handle_error(self, error: Exception, context: str = "",
                    show_dialog: bool = True, parent: Optional[tk.Tk] = None) -> bool:
        """Handle an error with logging and optional recovery"""
        error_info = self._format_error(error, context)

        # Log the error
        logger.error(f"Error in {context}: {error}")
        self._log_error_to_file(error_info)

        # Try recovery if available
        error_type = type(error).__name__
        if error_type in self.recovery_actions:
            try:
                logger.info(f"Attempting recovery for {error_type}")
                if self.recovery_actions[error_type](error, context):
                    logger.info(f"Recovery successful for {error_type}")
                    return True
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")

        # Show error dialog if requested
        if show_dialog and parent:
            self._show_error_dialog(parent, error, context)

        return False

    def _format_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Format error information for logging"""
        return {
            "timestamp": format_timestamp(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd()
            }
        }

    def _log_error_to_file(self, error_info: Dict[str, Any]):
        """Log error to file"""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
                f.write("\n---\n")
        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")

    def _show_error_dialog(self, parent: tk.Tk, error: Exception, context: str):
        """Show error dialog to user"""
        title = f"Error in {context}" if context else "Error"
        message = f"An error occurred:\n\n{str(error)}\n\nPlease check the logs for more details."

        messagebox.showerror(title, message, parent=parent)

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors from log file"""
        errors = []

        try:
            if not self.error_log_file.exists():
                return errors

            with open(self.error_log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by separator
            error_blocks = content.split("---\n")
            for block in error_blocks:
                if block.strip():
                    try:
                        error_info = json.loads(block.strip())
                        errors.append(error_info)
                    except json.JSONDecodeError:
                        continue

            # Sort by timestamp (most recent first) and limit
            errors.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return errors[:limit]

        except Exception as e:
            logger.error(f"Failed to read error log: {e}")
            return errors

    def clear_error_log(self):
        """Clear the error log file"""
        try:
            if self.error_log_file.exists():
                self.error_log_file.unlink()
                logger.info("Error log cleared")
        except Exception as e:
            logger.error(f"Failed to clear error log: {e}")

class RecoveryManager:
    """Manages recovery actions for different error scenarios"""

    def __init__(self):
        self.recovery_strategies: Dict[str, List[Callable]] = {}

    def add_recovery_strategy(self, error_type: str, strategy: Callable):
        """Add a recovery strategy for an error type"""
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        self.recovery_strategies[error_type].append(strategy)

    def attempt_recovery(self, error: Exception, context: str) -> bool:
        """Attempt to recover from an error"""
        error_type = type(error).__name__

        if error_type not in self.recovery_strategies:
            return False

        for strategy in self.recovery_strategies[error_type]:
            try:
                if strategy(error, context):
                    logger.info(f"Recovery strategy succeeded for {error_type}")
                    return True
            except Exception as strategy_error:
                logger.warning(f"Recovery strategy failed: {strategy_error}")
                continue

        return False

# Predefined recovery strategies
def recover_from_file_not_found(error: Exception, context: str) -> bool:
    """Recovery strategy for FileNotFoundError"""
    try:
        # Try to create missing directories
        if "datasets" in str(error).lower():
            from .dataset_manager import dataset_manager
            dataset_manager.datasets_dir.mkdir(parents=True, exist_ok=True)
            return True
        elif "models" in str(error).lower():
            from .training_manager import training_manager
            training_manager.models_dir.mkdir(parents=True, exist_ok=True)
            return True
    except Exception:
        pass
    return False

def recover_from_permission_error(error: Exception, context: str) -> bool:
    """Recovery strategy for PermissionError"""
    try:
        # Try to create directories with different permissions
        if "datasets" in str(error).lower():
            from .dataset_manager import dataset_manager
            os.makedirs(dataset_manager.datasets_dir, exist_ok=True)
            return True
        elif "models" in str(error).lower():
            from .training_manager import training_manager
            os.makedirs(training_manager.models_dir, exist_ok=True)
            return True
    except Exception:
        pass
    return False

def recover_from_yaml_error(error: Exception, context: str) -> bool:
    """Recovery strategy for YAML errors"""
    try:
        # Try to recreate data.yaml
        if "data.yaml" in str(error).lower():
            from .dataset_manager import dataset_manager
            if dataset_manager.current_dataset:
                dataset_path = dataset_manager.datasets_dir / dataset_manager.current_dataset
                data_yaml = dataset_path / "data.yaml"

                # Create basic data.yaml
                basic_config = {
                    "train": "images",
                    "val": "images",
                    "test": "images",
                    "nc": 1,
                    "names": ["object"]
                }

                import yaml
                with open(data_yaml, 'w') as f:
                    yaml.dump(basic_config, f, default_flow_style=False)

                logger.info(f"Recreated data.yaml for dataset: {dataset_manager.current_dataset}")
                return True
    except Exception:
        pass
    return False

# Global instances
error_handler = ErrorHandler()
recovery_manager = RecoveryManager()

# Register default recovery strategies
recovery_manager.add_recovery_strategy("FileNotFoundError", recover_from_file_not_found)
recovery_manager.add_recovery_strategy("PermissionError", recover_from_permission_error)
recovery_manager.add_recovery_strategy("YAMLError", recover_from_yaml_error)

# Register recovery actions with error handler
error_handler.register_recovery_action("FileNotFoundError", recovery_manager.attempt_recovery)
error_handler.register_recovery_action("PermissionError", recovery_manager.attempt_recovery)
error_handler.register_recovery_action("YAMLError", recovery_manager.attempt_recovery)

def safe_execute(func: Callable, *args, context: str = "", **kwargs) -> Any:
    """Execute a function safely with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, context)
        return None

def safe_execute_with_fallback(func: Callable, fallback: Any, *args,
                              context: str = "", **kwargs) -> Any:
    """Execute a function safely with a fallback value"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, context)
        return fallback

def log_operation_start(operation: str):
    """Log the start of an operation"""
    logger.info(f"Starting operation: {operation}")

def log_operation_end(operation: str, success: bool = True):
    """Log the end of an operation"""
    status = "completed successfully" if success else "failed"
    logger.info(f"Operation '{operation}' {status}")

# Context manager for operations
class OperationContext:
    """Context manager for operations with automatic logging and error handling"""

    def __init__(self, operation_name: str, log_start: bool = True):
        self.operation_name = operation_name
        self.log_start = log_start

    def __enter__(self):
        if self.log_start:
            log_operation_start(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            log_operation_end(self.operation_name, success=True)
        else:
            log_operation_end(self.operation_name, success=False)
            error_handler.handle_error(exc_val, self.operation_name)
            return False  # Re-raise the exception
