"""
Dataset management functionality for YOLO Trainer
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml

from . import logger, sanitize_filename, format_timestamp, ensure_directory

class DatasetManager:
    """Manages YOLO datasets"""

    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.current_dataset: Optional[str] = None
        ensure_directory(str(self.datasets_dir))

    def create_dataset(self, name: str, description: str = "") -> bool:
        """Create a new dataset"""
        try:
            # Sanitize name
            safe_name = sanitize_filename(name)
            dataset_path = self.datasets_dir / safe_name

            if dataset_path.exists():
                raise FileExistsError(f"Dataset '{safe_name}' already exists")

            # Create directory structure
            (dataset_path / "images").mkdir(parents=True)
            (dataset_path / "labels").mkdir()

            # Create data.yaml
            data_config = {
                "train": "images",
                "val": "images",
                "test": "images",
                "nc": 1,
                "names": ["object"]
            }

            with open(dataset_path / "data.yaml", 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)

            # Create metadata
            metadata = {
                "name": name,
                "safe_name": safe_name,
                "description": description,
                "created": format_timestamp(),
                "last_modified": format_timestamp(),
                "total_images": 0,
                "annotated_images": 0,
                "classes": ["object"],
                "class_count": 1
            }

            with open(dataset_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Created dataset: {safe_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create dataset '{name}': {e}")
            return False

    def load_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load dataset metadata"""
        try:
            dataset_path = self.datasets_dir / name
            metadata_file = dataset_path / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Update last accessed
            metadata["last_accessed"] = format_timestamp()
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.current_dataset = name
            return metadata

        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {e}")
            return None

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset"""
        try:
            dataset_path = self.datasets_dir / name

            if not dataset_path.exists():
                return False

            # Create backup before deletion
            backup_name = f"{name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.datasets_dir / "backups" / backup_name

            shutil.move(str(dataset_path), str(backup_path))
            logger.info(f"Dataset '{name}' moved to backup: {backup_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete dataset '{name}': {e}")
            return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        datasets = []

        try:
            for item in self.datasets_dir.iterdir():
                if item.is_dir() and item.name != "backups":
                    metadata_file = item / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            datasets.append(metadata)
                        except Exception as e:
                            logger.warning(f"Could not read metadata for {item.name}: {e}")

            # Sort by last modified (most recent first)
            datasets.sort(key=lambda x: x.get("last_modified", ""), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")

        return datasets

    def update_dataset_stats(self, name: str) -> bool:
        """Update dataset statistics"""
        try:
            dataset_path = self.datasets_dir / name
            metadata_file = dataset_path / "metadata.json"

            if not metadata_file.exists():
                return False

            # Count images and labels
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"

            total_images = 0
            annotated_images = 0

            if images_dir.exists():
                image_files = list(images_dir.glob("*"))
                total_images = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])

            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                annotated_images = len(label_files)

            # Update metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata["total_images"] = total_images
            metadata["annotated_images"] = annotated_images
            metadata["last_modified"] = format_timestamp()

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to update dataset stats for '{name}': {e}")
            return False

    def validate_dataset(self, name: str) -> Tuple[bool, List[str]]:
        """Validate dataset structure and files"""
        issues = []
        dataset_path = self.datasets_dir / name

        if not dataset_path.exists():
            return False, [f"Dataset directory does not exist: {dataset_path}"]

        # Check required directories
        required_dirs = ["images", "labels"]
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(f"Missing required directory: {dir_name}")

        # Check data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            issues.append("Missing data.yaml file")
        else:
            try:
                with open(data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)

                if not isinstance(data_config.get('nc'), int) or data_config['nc'] < 1:
                    issues.append("Invalid class count in data.yaml")

                if not isinstance(data_config.get('names'), list):
                    issues.append("Invalid class names in data.yaml")

            except Exception as e:
                issues.append(f"Error reading data.yaml: {e}")

        # Check for orphaned files
        if (dataset_path / "images").exists() and (dataset_path / "labels").exists():
            image_files = set()
            label_files = set()

            for img_file in (dataset_path / "images").glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.add(img_file.stem)

            for lbl_file in (dataset_path / "labels").glob("*.txt"):
                label_files.add(lbl_file.stem)

            orphaned_labels = label_files - image_files
            missing_labels = image_files - label_files

            if orphaned_labels:
                issues.append(f"Orphaned label files: {len(orphaned_labels)}")

            if missing_labels:
                issues.append(f"Images without labels: {len(missing_labels)}")

        return len(issues) == 0, issues

    def export_dataset(self, name: str, export_path: str) -> bool:
        """Export dataset to a compressed archive"""
        try:
            import zipfile

            dataset_path = self.datasets_dir / name
            if not dataset_path.exists():
                return False

            export_file = Path(export_path) / f"{name}_dataset.zip"

            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in dataset_path.rglob('*'):
                    if file_path.is_file():
                        arc_name = file_path.relative_to(dataset_path.parent)
                        zipf.write(file_path, arc_name)

            logger.info(f"Dataset exported to: {export_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export dataset '{name}': {e}")
            return False

    def import_dataset(self, archive_path: str, name: Optional[str] = None) -> bool:
        """Import dataset from archive"""
        try:
            import zipfile

            archive_file = Path(archive_path)
            if not archive_file.exists():
                return False

            # Extract dataset name from archive if not provided
            if name is None:
                name = archive_file.stem.replace('_dataset', '')

            if not name:
                name = f"imported_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            safe_name = sanitize_filename(name)
            dataset_path = self.datasets_dir / safe_name

            if dataset_path.exists():
                raise FileExistsError(f"Dataset '{safe_name}' already exists")

            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(dataset_path)

            # Validate imported dataset
            is_valid, issues = self.validate_dataset(safe_name)
            if not is_valid:
                logger.warning(f"Imported dataset has issues: {issues}")

            logger.info(f"Dataset imported: {safe_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to import dataset: {e}")
            return False

# Global dataset manager instance
dataset_manager = DatasetManager()
