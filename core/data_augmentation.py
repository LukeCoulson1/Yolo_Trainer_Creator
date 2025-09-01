"""
Data augmentation utilities for YOLO Trainer
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import random
import math

from . import logger, ensure_directory

class DataAugmentor:
    """Data augmentation utilities for YOLO datasets"""

    def __init__(self):
        self.augmentation_types = {
            "rotation": self._rotate_image,
            "flip": self._flip_image,
            "brightness": self._adjust_brightness,
            "contrast": self._adjust_contrast,
            "saturation": self._adjust_saturation,
            "hue": self._adjust_hue,
            "noise": self._add_noise,
            "blur": self._add_blur,
            "scale": self._scale_image,
            "crop": self._random_crop
        }

    def augment_dataset(self, dataset_path: str, output_path: str,
                       augmentations: Optional[Dict[str, Any]] = None,
                       num_augmentations: int = 1) -> bool:
        """Augment an entire dataset"""
        try:
            dataset_path_obj = Path(dataset_path)
            output_path_obj = Path(output_path)

            if not dataset_path_obj.exists():
                logger.error(f"Dataset path does not exist: {dataset_path_obj}")
                return False

            # Create output directories
            (output_path_obj / "images").mkdir(parents=True, exist_ok=True)
            (output_path_obj / "labels").mkdir(parents=True, exist_ok=True)

            # Default augmentations
            if augmentations is None:
                augmentations = {
                    "rotation": {"angle_range": (-15, 15)},
                    "flip": {"horizontal": True, "vertical": False},
                    "brightness": {"factor_range": (0.8, 1.2)},
                    "noise": {"intensity": 0.05}
                }

            # Process each image
            images_dir = dataset_path_obj / "images"
            labels_dir = dataset_path_obj / "labels"

            if not images_dir.exists():
                logger.error(f"Images directory not found: {images_dir}")
                return False

            processed_count = 0

            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Copy original
                    self._copy_original(img_file, output_path_obj / "images")

                    # Load corresponding label
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    labels = []
                    if label_file.exists():
                        labels = self._load_labels(label_file)
                        # Copy original label
                        self._copy_original(label_file, output_path_obj / "labels")

                    # Generate augmentations
                    for i in range(num_augmentations):
                        try:
                            augmented_img, augmented_labels = self._apply_augmentations(
                                img_file, labels, augmentations
                            )

                            if augmented_img is not None:
                                # Save augmented image
                                aug_img_name = f"{img_file.stem}_aug_{i}{img_file.suffix}"
                                aug_img_path = output_path_obj / "images" / aug_img_name
                                augmented_img.save(aug_img_path)

                                # Save augmented labels
                                if augmented_labels:
                                    aug_label_name = f"{img_file.stem}_aug_{i}.txt"
                                    aug_label_path = output_path_obj / "labels" / aug_label_name
                                    self._save_labels(aug_label_path, augmented_labels)

                                processed_count += 1

                        except Exception as e:
                            logger.warning(f"Failed to augment {img_file.name}: {e}")
                            continue

            logger.info(f"Dataset augmentation completed. Generated {processed_count} augmented images.")
            return True

        except Exception as e:
            logger.error(f"Failed to augment dataset: {e}")
            return False

    def _apply_augmentations(self, image_path: Path, labels: List[List[float]],
                           augmentations: Dict[str, Any]) -> Tuple[Optional[Image.Image], List[List[float]]]:
        """Apply augmentations to an image and its labels"""
        try:
            # Load image
            image = Image.open(image_path)
            img_array = np.array(image)

            # Apply each augmentation
            for aug_type, params in augmentations.items():
                if aug_type in self.augmentation_types:
                    img_array, labels = self.augmentation_types[aug_type](
                        img_array, labels, **params
                    )

            # Convert back to PIL Image
            augmented_image = Image.fromarray(img_array)
            return augmented_image, labels

        except Exception as e:
            logger.error(f"Failed to apply augmentations: {e}")
            return None, labels

    def _rotate_image(self, image: np.ndarray, labels: List[List[float]],
                     angle_range: Tuple[int, int] = (-15, 15)) -> Tuple[np.ndarray, List[List[float]]]:
        """Rotate image and adjust labels accordingly"""
        try:
            angle = random.uniform(*angle_range)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)

            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            # Adjust labels for rotation
            adjusted_labels = []
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = label[:5]

                    # Convert to absolute coordinates
                    abs_x = x_center * width
                    abs_y = y_center * height
                    abs_w = bbox_width * width
                    abs_h = bbox_height * height

                    # Apply rotation to center point
                    cos_a = math.cos(math.radians(-angle))
                    sin_a = math.sin(math.radians(-angle))

                    # Translate to center
                    x_translated = abs_x - center[0]
                    y_translated = abs_y - center[1]

                    # Rotate
                    x_rotated = x_translated * cos_a - y_translated * sin_a
                    y_rotated = x_translated * sin_a + y_translated * cos_a

                    # Translate back
                    x_final = x_rotated + center[0]
                    y_final = y_rotated + center[1]

                    # Convert back to relative coordinates
                    rel_x = x_final / width
                    rel_y = y_final / height

                    # Keep within bounds
                    rel_x = max(0, min(1, rel_x))
                    rel_y = max(0, min(1, rel_y))

                    adjusted_labels.append([class_id, rel_x, rel_y, bbox_width, bbox_height])

            return rotated_image, adjusted_labels

        except Exception as e:
            logger.warning(f"Rotation augmentation failed: {e}")
            return image, labels

    def _flip_image(self, image: np.ndarray, labels: List[List[float]],
                   horizontal: bool = True, vertical: bool = False) -> Tuple[np.ndarray, List[List[float]]]:
        """Flip image and adjust labels"""
        try:
            flip_code = -1 if horizontal and vertical else (1 if horizontal else 0)
            flipped_image = cv2.flip(image, flip_code)

            # Adjust labels for flip
            adjusted_labels = []
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = label[:5]

                    if horizontal:
                        x_center = 1.0 - x_center
                    if vertical:
                        y_center = 1.0 - y_center

                    adjusted_labels.append([class_id, x_center, y_center, bbox_width, bbox_height])

            return flipped_image, adjusted_labels

        except Exception as e:
            logger.warning(f"Flip augmentation failed: {e}")
            return image, labels

    def _adjust_brightness(self, image: np.ndarray, labels: List[List[float]],
                          factor_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust image brightness"""
        try:
            factor = random.uniform(*factor_range)
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            return adjusted_image, labels
        except Exception as e:
            logger.warning(f"Brightness adjustment failed: {e}")
            return image, labels

    def _adjust_contrast(self, image: np.ndarray, labels: List[List[float]],
                        factor_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust image contrast"""
        try:
            factor = random.uniform(*factor_range)
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            return adjusted_image, labels
        except Exception as e:
            logger.warning(f"Contrast adjustment failed: {e}")
            return image, labels

    def _adjust_saturation(self, image: np.ndarray, labels: List[List[float]],
                          factor_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust image saturation"""
        try:
            factor = random.uniform(*factor_range)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return adjusted_image, labels
        except Exception as e:
            logger.warning(f"Saturation adjustment failed: {e}")
            return image, labels

    def _adjust_hue(self, image: np.ndarray, labels: List[List[float]],
                   shift_range: Tuple[int, int] = (-10, 10)) -> Tuple[np.ndarray, List[List[float]]]:
        """Adjust image hue"""
        try:
            shift = random.randint(*shift_range)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return adjusted_image, labels
        except Exception as e:
            logger.warning(f"Hue adjustment failed: {e}")
            return image, labels

    def _add_noise(self, image: np.ndarray, labels: List[List[float]],
                  intensity: float = 0.05) -> Tuple[np.ndarray, List[List[float]]]:
        """Add random noise to image"""
        try:
            noise = np.random.normal(0, intensity * 255, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
            return noisy_image, labels
        except Exception as e:
            logger.warning(f"Noise addition failed: {e}")
            return image, labels

    def _add_blur(self, image: np.ndarray, labels: List[List[float]],
                 kernel_size: int = 3) -> Tuple[np.ndarray, List[List[float]]]:
        """Add blur to image"""
        try:
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return blurred_image, labels
        except Exception as e:
            logger.warning(f"Blur addition failed: {e}")
            return image, labels

    def _scale_image(self, image: np.ndarray, labels: List[List[float]],
                    scale_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, List[List[float]]]:
        """Scale image and adjust labels"""
        try:
            scale = random.uniform(*scale_range)
            height, width = image.shape[:2]

            new_width = int(width * scale)
            new_height = int(height * scale)

            scaled_image = cv2.resize(image, (new_width, new_height))

            # Adjust labels for scaling
            adjusted_labels = []
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = label[:5]
                    # Labels remain the same since we're using relative coordinates
                    adjusted_labels.append([class_id, x_center, y_center, bbox_width, bbox_height])

            return scaled_image, adjusted_labels

        except Exception as e:
            logger.warning(f"Scale augmentation failed: {e}")
            return image, labels

    def _random_crop(self, image: np.ndarray, labels: List[List[float]],
                    crop_factor: float = 0.9) -> Tuple[np.ndarray, List[List[float]]]:
        """Random crop of image"""
        try:
            height, width = image.shape[:2]

            crop_height = int(height * crop_factor)
            crop_width = int(width * crop_factor)

            # Random crop position
            x_start = random.randint(0, width - crop_width)
            y_start = random.randint(0, height - crop_height)

            cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

            # Adjust labels for crop
            adjusted_labels = []
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = label[:5]

                    # Convert to absolute coordinates
                    abs_x = x_center * width
                    abs_y = y_center * height

                    # Check if bbox is within crop area
                    if (x_start <= abs_x <= x_start + crop_width and
                        y_start <= abs_y <= y_start + crop_height):

                        # Adjust coordinates relative to crop
                        new_x = (abs_x - x_start) / crop_width
                        new_y = (abs_y - y_start) / crop_height

                        # Keep within bounds
                        new_x = max(0, min(1, new_x))
                        new_y = max(0, min(1, new_y))

                        adjusted_labels.append([class_id, new_x, new_y, bbox_width, bbox_height])

            return cropped_image, adjusted_labels

        except Exception as e:
            logger.warning(f"Crop augmentation failed: {e}")
            return image, labels

    def _load_labels(self, label_file: Path) -> List[List[float]]:
        """Load YOLO labels from file"""
        labels = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(p) for p in parts])
        except Exception as e:
            logger.warning(f"Failed to load labels from {label_file}: {e}")
        return labels

    def _save_labels(self, label_file: Path, labels: List[List[float]]):
        """Save YOLO labels to file"""
        try:
            with open(label_file, 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save labels to {label_file}: {e}")

    def _copy_original(self, source: Path, dest_dir: Path):
        """Copy original file to destination directory"""
        try:
            import shutil
            shutil.copy2(source, dest_dir / source.name)
        except Exception as e:
            logger.warning(f"Failed to copy {source} to {dest_dir}: {e}")

# Global data augmentor instance
data_augmentor = DataAugmentor()
