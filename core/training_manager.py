"""
Training management functionality for YOLO Trainer
"""

import os
import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import yaml

from . import logger, sanitize_filename, format_timestamp, ensure_directory

class TrainingManager:
    """Manages YOLO training operations"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.current_training: Optional[Dict[str, Any]] = None
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        self.training_callbacks: List[Callable] = []
        ensure_directory(str(self.models_dir))

    def add_training_callback(self, callback: Callable):
        """Add callback for training progress updates"""
        self.training_callbacks.append(callback)

    def remove_training_callback(self, callback: Callable):
        """Remove training callback"""
        if callback in self.training_callbacks:
            self.training_callbacks.remove(callback)

    def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """Notify all registered callbacks"""
        for callback in self.training_callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Training callback error: {e}")

    def start_training(self, config: Dict[str, Any]) -> bool:
        """Start YOLO training in background thread"""
        if self.is_training:
            logger.warning("Training already in progress")
            return False

        try:
            # Validate configuration
            if not self._validate_training_config(config):
                return False

            # Create model directory
            model_name = config.get('model_name', f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            safe_name = sanitize_filename(model_name)
            model_dir = self.models_dir / safe_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save training configuration
            config_file = model_dir / "training_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(config, model_dir),
                daemon=True
            )
            self.training_thread.start()

            self.is_training = True
            self.current_training = {
                "model_name": model_name,
                "safe_name": safe_name,
                "model_dir": str(model_dir),
                "start_time": format_timestamp(),
                "status": "running",
                "config": config
            }

            self._notify_callbacks("training_started", self.current_training)
            logger.info(f"Started training: {model_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    def stop_training(self) -> bool:
        """Stop current training"""
        if not self.is_training:
            return False

        try:
            self.is_training = False
            if self.training_thread and self.training_thread.is_alive():
                # Note: In a real implementation, you'd need to handle graceful termination
                # This is a simplified version
                pass

            if self.current_training:
                self.current_training["status"] = "stopped"
                self.current_training["end_time"] = format_timestamp()
                self._notify_callbacks("training_stopped", self.current_training)

            logger.info("Training stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return False

    def _run_training(self, config: Dict[str, Any], model_dir: Path):
        """Execute training process"""
        try:
            from ultralytics import YOLO

            # Load model
            model_path = config.get('model_path', 'yolov8n.pt')
            model = YOLO(model_path)

            # Prepare training arguments
            train_args = {
                'data': config['data_yaml'],
                'epochs': config.get('epochs', 100),
                'batch': config.get('batch_size', 16),
                'imgsz': config.get('img_size', 640),
                'project': str(model_dir.parent),
                'name': model_dir.name,
                'save': True,
                'save_period': config.get('save_period', 10),
                'cache': config.get('cache', False),
                'device': config.get('device', 0),
                'workers': config.get('workers', 8),
                'patience': config.get('patience', 50),
                'cos_lr': config.get('cos_lr', False),
                'close_mosaic': config.get('close_mosaic', 10),
                'resume': config.get('resume', False)
            }

            # Add optional arguments
            if 'lr0' in config:
                train_args['lr0'] = config['lr0']
            if 'lrf' in config:
                train_args['lrf'] = config['lrf']
            if 'momentum' in config:
                train_args['momentum'] = config['momentum']
            if 'weight_decay' in config:
                train_args['weight_decay'] = config['weight_decay']

            # Start training
            self._notify_callbacks("training_progress", {"message": "Starting training..."})

            results = model.train(**train_args)

            # Training completed
            if self.current_training:
                self.current_training["status"] = "completed"
                self.current_training["end_time"] = format_timestamp()
                self.current_training["results"] = {
                    "best_fitness": float(results.best_fitness) if results and hasattr(results, 'best_fitness') else 0.0,
                    "best_model_path": str(results.save_dir / "weights" / "best.pt") if results and hasattr(results, 'save_dir') else ""
                }

            if self.current_training:
                self._notify_callbacks("training_completed", self.current_training)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.current_training:
                self.current_training["status"] = "failed"
                self.current_training["error"] = str(e)
                self.current_training["end_time"] = format_timestamp()
                self._notify_callbacks("training_failed", self.current_training)

        finally:
            self.is_training = False

    def _validate_training_config(self, config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        required_fields = ['data_yaml']
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False

        # Check if data.yaml exists
        data_yaml = Path(config['data_yaml'])
        if not data_yaml.exists():
            logger.error(f"Data YAML file not found: {data_yaml}")
            return False

        # Validate numeric fields
        numeric_fields = ['epochs', 'batch_size', 'img_size', 'save_period', 'workers', 'patience']
        for field in numeric_fields:
            if field in config and not isinstance(config[field], (int, float)):
                logger.error(f"Invalid {field}: must be numeric")
                return False

        return True

    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """Get current training status"""
        return self.current_training

    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models"""
        models = []

        try:
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    config_file = item / "training_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)

                            # Get model info
                            weights_dir = item / "weights"
                            best_model = weights_dir / "best.pt" if weights_dir.exists() else None
                            last_model = weights_dir / "last.pt" if weights_dir.exists() else None

                            model_info = {
                                "name": config.get('model_name', item.name),
                                "safe_name": item.name,
                                "created": config.get('created', ''),
                                "dataset": config.get('dataset', ''),
                                "epochs": config.get('epochs', 0),
                                "best_fitness": config.get('best_fitness', 0.0),
                                "has_best_model": best_model.exists() if best_model else False,
                                "has_last_model": last_model.exists() if last_model else False,
                                "model_dir": str(item)
                            }
                            models.append(model_info)

                        except Exception as e:
                            logger.warning(f"Could not read config for {item.name}: {e}")

            # Sort by creation date (most recent first)
            models.sort(key=lambda x: x.get("created", ""), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list trained models: {e}")

        return models

    def delete_model(self, model_name: str) -> bool:
        """Delete a trained model"""
        try:
            model_dir = self.models_dir / model_name

            if not model_dir.exists():
                return False

            # Create backup before deletion
            backup_name = f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.models_dir / "backups" / backup_name

            import shutil
            shutil.move(str(model_dir), str(backup_path))
            logger.info(f"Model '{model_name}' moved to backup: {backup_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete model '{model_name}': {e}")
            return False

    def export_model(self, model_name: str, export_path: str) -> bool:
        """Export trained model"""
        try:
            model_dir = self.models_dir / model_name
            if not model_dir.exists():
                return False

            import shutil
            export_file = Path(export_path) / f"{model_name}_model.zip"

            # Create zip archive
            shutil.make_archive(str(export_file.with_suffix('')), 'zip', model_dir)

            logger.info(f"Model exported to: {export_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export model '{model_name}': {e}")
            return False

# Global training manager instance
training_manager = TrainingManager()
