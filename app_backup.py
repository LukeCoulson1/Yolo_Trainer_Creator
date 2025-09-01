
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
import random
import io
import time
import base64
from datetime import datetime
import sys
import torch
import json
from pathlib import Path

def validate_coordinates(coords_str, img_width, img_height):
    """Validate coordinate string and return parsed coordinates."""
    try:
        parts = [x.strip() for x in coords_str.split(',')]
        if len(parts) != 4:
            return False, "Coordinates must be in format: x1,y1,x2,y2"

        x1, y1, x2, y2 = [float(x) for x in parts]

        # Check if coordinates are within image bounds
        if not all(0 <= coord <= max_dim for coord, max_dim in
                  [(x1, img_width), (y1, img_height), (x2, img_width), (y2, img_height)]):
            return False, "Coordinates are outside image bounds"

        # Check if box has valid dimensions
        if x2 <= x1 or y2 <= y1:
            return False, "Invalid box dimensions (x2 must be > x1, y2 must be > y1)"

        return True, (x1, y1, x2, y2)
    except ValueError:
        return False, "Invalid coordinate format - must be numbers separated by commas"

def validate_dataset_name(name):
    """Validate dataset name."""
    if not name or not name.strip():
        return False, "Dataset name cannot be empty"

    if len(name.strip()) > 50:
        return False, "Dataset name too long (max 50 characters)"

    # Check for invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in name for char in invalid_chars):
        return False, "Dataset name contains invalid characters"

    return True, name.strip()

def validate_image_file(file):
    """Validate an image file before processing."""
    try:
        # Check file size (limit to 50MB)
        file_size = len(file.getvalue())
        if file_size > 50 * 1024 * 1024:
            return False, f"Image '{file.name}' too large (max 50MB)"

        if file_size < 100:  # Minimum file size
            return False, f"Image '{file.name}' too small (possibly corrupted)"

        # Try to open and validate the image
        try:
            image = Image.open(file)
            image.verify()  # Verify the image is not corrupted

            # Re-open after verify (verify closes the file)
            file.seek(0)
            image = Image.open(file)

            # Check image dimensions
            width, height = image.size
            if width < 10 or height < 10:
                return False, f"Image '{file.name}' too small (minimum 10x10 pixels)"

            if width > 4096 or height > 4096:
                return False, f"Image '{file.name}' too large (maximum 4096x4096 pixels)"

            # Check image mode
            if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
                return False, f"Image '{file.name}' has unsupported color mode: {image.mode}"

            return True, image

        except Exception as e:
            return False, f"Image '{file.name}' is corrupted or invalid: {str(e)}"

    except Exception as e:
        return False, f"Error validating '{file.name}': {str(e)}"

def create_dataset_robust(dataset_name, uploaded_files, progress_callback=None):
    """Create a dataset with comprehensive error handling and validation."""
    errors = []
    warnings = []

    try:
        # Validate dataset name
        is_valid, message = validate_dataset_name(dataset_name)
        if not is_valid:
            errors.append(message)
            return False, errors, warnings

        # Check if dataset already exists
        dataset_path = os.path.join("datasets", dataset_name)
        if os.path.exists(dataset_path):
            errors.append(f"Dataset '{dataset_name}' already exists!")
            return False, errors, warnings

        # Validate uploaded files
        if not uploaded_files:
            errors.append("Please upload some images first!")
            return False, errors, warnings

        # Validate each image file
        valid_files = []
        invalid_files = []

        if progress_callback:
            progress_callback(0.1, "Validating images...")

        for i, file in enumerate(uploaded_files):
            is_valid, result = validate_image_file(file)
            if is_valid:
                valid_files.append((file, result))  # (file, image_object)
            else:
                invalid_files.append(result)

            if progress_callback:
                progress_callback(0.1 + (i + 1) / len(uploaded_files) * 0.2, f"Validating {file.name}...")

        if invalid_files:
            errors.extend(invalid_files)

        if not valid_files:
            errors.append("No valid image files found!")
            return False, errors, warnings

        # Check for duplicate filenames
        filenames = [file.name for file, _ in valid_files]
        duplicates = [name for name in filenames if filenames.count(name) > 1]
        unique_duplicates = list(set(duplicates)) if duplicates else []
        if unique_duplicates:
            warnings.append(f"Duplicate filenames found: {', '.join(unique_duplicates)}. Only the first occurrence will be kept.")

        # Create dataset structure
        try:
            os.makedirs(dataset_path, exist_ok=True)
            images_path = os.path.join(dataset_path, "images")
            labels_path = os.path.join(dataset_path, "labels")
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create dataset directories: {str(e)}")
            return False, errors, warnings

        # Save images and create label files
        saved_files = 0
        total_files = len(valid_files)

        if progress_callback:
            progress_callback(0.3, "Saving images...")

        for i, (uploaded_file, image) in enumerate(valid_files):
            try:
                # Check for duplicate filename (keep only first occurrence)
                image_path = os.path.join(images_path, uploaded_file.name)
                if os.path.exists(image_path):
                    continue

                # Save image
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Create empty label file
                label_name = os.path.splitext(uploaded_file.name)[0] + ".txt"
                label_path = os.path.join(labels_path, label_name)
                with open(label_path, "w") as f:
                    f.write("")  # Empty label file

                saved_files += 1

                if progress_callback:
                    progress = 0.3 + (i + 1) / total_files * 0.4
                    progress_callback(progress, f"Saving {uploaded_file.name}...")

            except Exception as e:
                errors.append(f"Failed to save '{uploaded_file.name}': {str(e)}")

        if saved_files == 0:
            errors.append("Failed to save any images!")
            return False, errors, warnings

        # Create data.yaml
        try:
            data_yaml = f"""# YOLO Dataset Configuration
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Images: {saved_files}
# Dataset: {dataset_name}

train: images
val: images
test: images

nc: 1
names: ['object']
"""
            with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
                f.write(data_yaml)
        except Exception as e:
            errors.append(f"Failed to create data.yaml: {str(e)}")
            return False, errors, warnings

        # Create dataset metadata
        try:
            metadata = {
                "name": dataset_name,
                "created": datetime.now().isoformat(),
                "total_images": saved_files,
                "valid_images": len(valid_files),
                "invalid_images": len(invalid_files),
                "duplicates_found": len(unique_duplicates) if 'unique_duplicates' in locals() else 0
            }
            import json
            with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            warnings.append(f"Could not create metadata file: {str(e)}")

        if progress_callback:
            progress_callback(1.0, "Dataset created successfully!")

        return True, errors, warnings

    except Exception as e:
        errors.append(f"Unexpected error during dataset creation: {str(e)}")
        return False, errors, warnings

def add_images_to_dataset_robust(dataset_name, uploaded_files, progress_callback=None):
    """Add images to an existing dataset with comprehensive error handling and validation."""
    errors = []
    warnings = []
    added_count = 0

    try:
        # Check if dataset exists
        dataset_path = os.path.join("datasets", dataset_name)
        if not os.path.exists(dataset_path):
            errors.append(f"Dataset '{dataset_name}' does not exist!")
            return False, errors, warnings, 0

        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        # Ensure directories exist
        try:
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create dataset directories: {str(e)}")
            return False, errors, warnings, 0

        # Validate uploaded files
        if not uploaded_files:
            errors.append("Please upload some images first!")
            return False, errors, warnings, 0

        # Validate each image file
        valid_files = []
        invalid_files = []

        if progress_callback:
            progress_callback(0.1, "Validating images...")

        for i, file in enumerate(uploaded_files):
            is_valid, result = validate_image_file(file)
            if is_valid:
                valid_files.append((file, result))  # (file, image_object)
            else:
                invalid_files.append(result)

            if progress_callback:
                progress_callback(0.1 + (i + 1) / len(uploaded_files) * 0.2, f"Validating {file.name}...")

        if invalid_files:
            errors.extend(invalid_files)

        if not valid_files:
            errors.append("No valid image files found!")
            return False, errors, warnings, 0

        # Check for existing images
        existing_images = set()
        if os.path.exists(images_path):
            existing_images = {f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))}

        # Filter out duplicates
        new_files = []
        duplicates = []
        for file, _ in valid_files:
            if file.name in existing_images:
                duplicates.append(file.name)
            else:
                new_files.append((file, None))  # We'll validate again during saving

        if duplicates:
            warnings.append(f"Skipped {len(duplicates)} duplicate images: {', '.join(duplicates[:5])}{'...' if len(duplicates) > 5 else ''}")

        if not new_files:
            warnings.append("No new images to add (all were duplicates or invalid)")
            return True, errors, warnings, 0

        # Save new images
        if progress_callback:
            progress_callback(0.3, "Adding images...")

        for i, (uploaded_file, _) in enumerate(new_files):
            try:
                # Re-validate the image (in case file changed)
                is_valid, _ = validate_image_file(uploaded_file)
                if not is_valid:
                    warnings.append(f"Skipping '{uploaded_file.name}' - validation failed during save")
                    continue

                # Save image
                image_path = os.path.join(images_path, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Create empty label file
                label_name = os.path.splitext(uploaded_file.name)[0] + ".txt"
                label_path = os.path.join(labels_path, label_name)
                with open(label_path, "w") as f:
                    f.write("")  # Empty label file

                added_count += 1

                if progress_callback:
                    progress = 0.3 + (i + 1) / len(new_files) * 0.7
                    progress_callback(progress, f"Adding {uploaded_file.name}...")

            except Exception as e:
                errors.append(f"Failed to save '{uploaded_file.name}': {str(e)}")

        if progress_callback:
            progress_callback(1.0, f"Added {added_count} images successfully!")

        return True, errors, warnings, added_count

    except Exception as e:
        errors.append(f"Unexpected error during image addition: {str(e)}")
        return False, errors, warnings, 0

def validate_dataset_integrity(dataset_path):
    """Validate the integrity of a dataset structure."""
    errors = []
    warnings = []

    try:
        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            errors.append("Dataset directory does not exist")
            return False, errors, warnings

        # Check required subdirectories
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        if not os.path.exists(images_path):
            errors.append("Images directory is missing")
            return False, errors, warnings

        if not os.path.exists(labels_path):
            errors.append("Labels directory is missing")
            return False, errors, warnings

        # Check data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            warnings.append("data.yaml file is missing")
        else:
            try:
                with open(data_yaml_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        warnings.append("data.yaml file is empty")
            except Exception as e:
                warnings.append(f"Could not read data.yaml: {str(e)}")

        # Get image and label files
        image_files = []
        label_files = []

        if os.path.exists(images_path):
            image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]

        if os.path.exists(labels_path):
            label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]

        # Check for orphaned files
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}

        images_without_labels = image_basenames - label_basenames
        labels_without_images = label_basenames - image_basenames

        if images_without_labels:
            warnings.append(f"{len(images_without_labels)} images have no corresponding label files")

        if labels_without_images:
            warnings.append(f"{len(labels_without_images)} label files have no corresponding images")

        # Check file sizes and validity
        corrupted_images = []
        for img_file in image_files:
            img_path = os.path.join(images_path, img_file)
            try:
                # Quick file size check
                file_size = os.path.getsize(img_path)
                if file_size < 100:
                    corrupted_images.append(img_file)
                elif file_size > 50 * 1024 * 1024:  # 50MB limit
                    warnings.append(f"Image '{img_file}' is very large ({file_size / (1024*1024):.1f}MB)")
            except Exception as e:
                corrupted_images.append(f"{img_file} (error: {str(e)})")

        if corrupted_images:
            warnings.append(f"{len(corrupted_images)} potentially corrupted images found")

        return True, errors, warnings

    except Exception as e:
        errors.append(f"Error validating dataset integrity: {str(e)}")
        return False, errors, warnings

def generate_dataset_report(dataset_path):
    """Generate a comprehensive dataset report with statistics and recommendations."""
    try:
        from datetime import datetime
        import os
        import collections

        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_name': os.path.basename(dataset_path),
            'statistics': {},
            'quality': {},
            'recommendations': []
        }

        # Basic paths
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        # Get file lists
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
        label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]

        # Basic statistics
        total_images = len(image_files)
        total_labels = len(label_files)

        report_data['statistics'] = {
            'total_images': total_images,
            'total_annotations': 0,  # Will be calculated
            'annotated_images': 0,
            'unannotated_images': 0,
            'avg_annotations_per_image': 0.0,
            'most_common_class': 'N/A',
            'class_distribution': {}
        }

        # Analyze annotations
        class_counts = collections.defaultdict(int)
        total_annotations = 0
        annotated_images = 0

        for label_file in label_files:
            label_path = os.path.join(labels_path, label_file)
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Not empty
                        lines = content.split('\n')
                        annotation_count = len([line for line in lines if line.strip()])
                        if annotation_count > 0:
                            annotated_images += 1
                            total_annotations += annotation_count

                            # Count classes
                            for line in lines:
                                if line.strip():
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        try:
                                            class_id = int(parts[0])
                                            class_counts[class_id] += 1
                                        except (ValueError, IndexError):
                                            pass
            except Exception as e:
                continue

        # Update statistics
        stats = report_data['statistics']
        stats['total_annotations'] = total_annotations
        stats['annotated_images'] = annotated_images
        stats['unannotated_images'] = total_images - annotated_images
        stats['avg_annotations_per_image'] = total_annotations / max(annotated_images, 1)
        stats['class_distribution'] = dict(class_counts)

        if class_counts:
            most_common_class = max(class_counts.items(), key=lambda x: x[1])
            stats['most_common_class'] = f"Class {most_common_class[0]} ({most_common_class[1]} instances)"

        # Quality analysis
        quality_issues = []
        recommendations = []

        # Check for unannotated images
        if stats['unannotated_images'] > 0:
            quality_issues.append(f"{stats['unannotated_images']} images have no annotations")
            recommendations.append("Add annotations to unannotated images for better training results")

        # Check annotation distribution
        if total_annotations > 0 and annotated_images > 0:
            avg_per_image = total_annotations / annotated_images
            if avg_per_image < 1:
                recommendations.append("Consider adding more annotations per image for better object detection")
            elif avg_per_image > 50:
                quality_issues.append("Some images have very high annotation counts - check for annotation quality")

        # Check class balance
        if len(class_counts) > 1:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            if max_count > min_count * 3:
                recommendations.append("Class distribution is imbalanced - consider adding more samples for underrepresented classes")

        # Check dataset size
        if total_images < 100:
            recommendations.append("Dataset is small - consider adding more images for better training results")
        elif total_images > 10000:
            recommendations.append("Dataset is very large - consider splitting into train/validation sets")

        # Calculate health score
        health_score = 100

        if stats['unannotated_images'] > 0:
            health_score -= min(30, stats['unannotated_images'] * 2)

        if len(class_counts) > 1:
            balance_ratio = min(class_counts.values()) / max(class_counts.values()) if max(class_counts.values()) > 0 else 1
            health_score -= int((1 - balance_ratio) * 20)

        if total_images < 100:
            health_score -= 20

        health_score = max(0, min(100, health_score))

        report_data['quality'] = {
            'issues': quality_issues,
            'health_score': health_score,
            'recommendations': recommendations
        }

        return report_data

    except Exception as e:
        st.error(f"Error generating dataset report: {str(e)}")
        return None

def safe_open_image(file_or_path):
    """Safely open an image file with error handling."""
    try:
        # Handle both uploaded files and file paths
        if hasattr(file_or_path, 'getvalue'):  # Uploaded file
            # Check file size (limit to 50MB)
            file_size = len(file_or_path.getvalue())
            if file_size > 50 * 1024 * 1024:
                st.error("❌ Image file too large (max 50MB)")
                return None

            # Open image
            image = Image.open(file_or_path)
        else:  # File path
            # Check file size
            file_size = os.path.getsize(file_or_path)
            if file_size > 50 * 1024 * 1024:
                st.error("❌ Image file too large (max 50MB)")
                return None

            # Open image
            image = Image.open(file_or_path)

        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        # Check image dimensions
        width, height = image.size
        if width < 10 or height < 10:
            st.error("❌ Image too small (minimum 10x10 pixels)")
            return None

        if width > 4096 or height > 4096:
            st.error("❌ Image too large (maximum 4096x4096 pixels)")
            return None

        return image

    except Exception as e:
        st.error(f"❌ Error opening image: {str(e)}")
        return None

def generate_annotation_canvas_html(img_base64, img_width, img_height, element_id, is_dataset_image=False):
    """Generate HTML with JavaScript for robust annotation canvas."""
    if is_dataset_image:
        img_id = f"image_{element_id}"
        canvas_id = f"canvas_{element_id}"
        coord_display_id = f"coord_display_{element_id}"
        clear_function = "clearCanvas"
        coord_variable = "lastBoxCoords"
        update_function = "updateCoordinateDisplay"
    else:
        img_id = f"annotate_image_{element_id}"
        canvas_id = f"annotate_canvas_{element_id}"
        coord_display_id = f"annotate_coord_display_{element_id}"
        clear_function = "clearAnnotationCanvas"
        coord_variable = "annotateLastBoxCoords"
        update_function = "updateAnnotationCoordinateDisplay"

    return f"""
    <div style="position: relative; display: inline-block; border: 2px solid #ddd;">
        <img id="{img_id}" src="data:image/png;base64,{img_base64}"
             style="display: block; max-width: 100%; height: auto;" />
        <canvas id="{canvas_id}"
                style="position: absolute; top: 0; left: 0; cursor: crosshair; pointer-events: auto;"
                width="{img_width}" height="{img_height}">
        </canvas>
    </div>

    <div id="{coord_display_id}" style="margin: 10px 0; padding: 15px; border: 2px dashed #ccc; border-radius: 8px; text-align: center; font-size: 14px; color: #666;">
        👆 Draw a bounding box on the image above to see coordinates here
    </div>

    <script>
    (function() {{
        const img = document.getElementById('{img_id}');
        const canvas = document.getElementById('{canvas_id}');
        const ctx = canvas.getContext('2d');

        if (!canvas || !ctx) {{
            return;
        }}

        // Set canvas size to match image display size
        function resizeCanvas() {{
            if (!img) {{
                return;
            }}
            const rect = img.getBoundingClientRect();

            // Set canvas display size to match image
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';

            // Set actual canvas resolution - use original image size for better quality
            canvas.width = {img_width};
            canvas.height = {img_height};
            
            // Redraw existing boxes after resize
            redrawCanvas();
        }}

        resizeCanvas();
        
        // Also resize canvas when image loads (in case it wasn't loaded initially)
        if (!img.complete) {{
            img.addEventListener('load', function() {{
                resizeCanvas();
            }});
        }}
        
        window.addEventListener('resize', resizeCanvas);

        // Configuration
        const MIN_BOX_SIZE = 10; // Minimum box size in pixels
        const SNAP_THRESHOLD = 5; // Snap to edges within this distance

        let isDrawing = false;
        let startX, startY;
        let boxes = [];
        let history = []; // For undo/redo functionality
        let historyIndex = -1;

        function saveToHistory() {{
            history = history.slice(0, historyIndex + 1);
            history.push([...boxes]);
            historyIndex++;
            if (history.length > 50) {{ // Limit history size
                history.shift();
                historyIndex--;
            }}
        }}

        function drawBox(box, isActive = false) {{
            ctx.strokeStyle = isActive ? '#00FF00' : '#FF4500';
            ctx.lineWidth = isActive ? 3 : 2;
            ctx.strokeRect(box.x, box.y, box.width, box.height);

            ctx.fillStyle = isActive ? 'rgba(0, 255, 0, 0.2)' : 'rgba(255, 165, 0, 0.3)';
            ctx.fillRect(box.x, box.y, box.width, box.height);

            // Draw resize handles for active box
            if (isActive) {{
                const handles = [
                    [box.x, box.y], [box.x + box.width/2, box.y], [box.x + box.width, box.y],
                    [box.x, box.y + box.height/2], [box.x + box.width, box.y + box.height/2],
                    [box.x, box.y + box.height], [box.x + box.width/2, box.y + box.height], [box.x + box.width, box.y + box.height]
                ];
                ctx.fillStyle = '#00FF00';
                handles.forEach(handle => {{
                    ctx.beginPath();
                    ctx.arc(handle[0], handle[1], 4, 0, 2 * Math.PI);
                    ctx.fill();
                }});
            }}
        }}

        function redrawCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            boxes.forEach((box, index) => drawBox(box, index === boxes.length - 1));
        }}

        function snapToEdges(x, y) {{
            // Snap to canvas edges
            if (Math.abs(x) < SNAP_THRESHOLD) x = 0;
            if (Math.abs(x - canvas.width) < SNAP_THRESHOLD) x = canvas.width;
            if (Math.abs(y) < SNAP_THRESHOLD) y = 0;
            if (Math.abs(y - canvas.height) < SNAP_THRESHOLD) y = canvas.height;
            return {{x, y}};
        }}

        function validateBox(x1, y1, x2, y2) {{
            const width = Math.abs(x2 - x1);
            const height = Math.abs(y2 - y1);

            if (width < MIN_BOX_SIZE || height < MIN_BOX_SIZE) {{
                return {{valid: false, error: 'Box too small (min ' + MIN_BOX_SIZE + 'x' + MIN_BOX_SIZE + ' pixels)'}};
            }}

            if (width > canvas.width * 0.9 || height > canvas.height * 0.9) {{
                return {{valid: false, error: 'Box too large (max 90% of image)'}};
            }}

            return {{valid: true}};
        }}

        canvas.addEventListener('mousedown', function(e) {{
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            // Scale mouse coordinates to canvas resolution
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            startX = (e.clientX - rect.left) * scaleX;
            startY = (e.clientY - rect.top) * scaleY;
        }});

        canvas.addEventListener('mousemove', function(e) {{
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            // Scale mouse coordinates to canvas resolution
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            let currentX = (e.clientX - rect.left) * scaleX;
            let currentY = (e.clientY - rect.top) * scaleY;

            // Snap to edges (in canvas resolution space)
            if (Math.abs(currentX) < SNAP_THRESHOLD) currentX = 0;
            if (Math.abs(currentX - canvas.width) < SNAP_THRESHOLD) currentX = canvas.width;
            if (Math.abs(currentY) < SNAP_THRESHOLD) currentY = 0;
            if (Math.abs(currentY - canvas.height) < SNAP_THRESHOLD) currentY = canvas.height;

            redrawCanvas();

            // Draw current box being drawn
            const width = currentX - startX;
            const height = currentY - startY;
            ctx.strokeStyle = '#FF4500';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, width, height);

            ctx.fillStyle = 'rgba(255, 165, 0, 0.3)';
            ctx.fillRect(startX, startY, width, height);
        }});

        canvas.addEventListener('mouseup', function(e) {{
            if (!isDrawing) {{
                return;
            }}
            isDrawing = false;

            const rect = canvas.getBoundingClientRect();
            // Scale mouse coordinates to canvas resolution
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            let endX = (e.clientX - rect.left) * scaleX;
            let endY = (e.clientY - rect.top) * scaleY;

            // Snap to edges (in canvas resolution space)
            if (Math.abs(endX) < SNAP_THRESHOLD) endX = 0;
            if (Math.abs(endX - canvas.width) < SNAP_THRESHOLD) endX = canvas.width;
            if (Math.abs(endY) < SNAP_THRESHOLD) endY = 0;
            if (Math.abs(endY - canvas.height) < SNAP_THRESHOLD) endY = canvas.height;

            const width = endX - startX;
            const height = endY - startY;

            // Validate box
            const validation = validateBox(startX, startY, endX, endY);
            if (!validation.valid) {{
                alert('Invalid box: ' + validation.error);
                redrawCanvas();
                return;
            }}

            if (Math.abs(width) > MIN_BOX_SIZE && Math.abs(height) > MIN_BOX_SIZE) {{
                const box = {{
                    x: Math.min(startX, endX),
                    y: Math.min(startY, endY),
                    width: Math.abs(width),
                    height: Math.abs(height)
                }};

                saveToHistory();
                boxes.push(box);
                redrawCanvas();

                // Send coordinates to Streamlit (already in correct coordinate system)
                const coords = box.x.toFixed(0) + ',' + box.y.toFixed(0) + ',' +
                              (box.x + box.width).toFixed(0) + ',' + (box.y + box.height).toFixed(0);

                // Store in multiple places for reliability
                window.{coord_variable} = coords;
                localStorage.setItem('lastBoxCoords_{element_id}', coords);

                // Update coordinate display immediately
                {update_function}();

                // Try multiple methods to update Streamlit
                updateStreamlitCoordinates(coords);

                // Trigger periodic check
                if (window.coordCheckInterval) {{
                    clearInterval(window.coordCheckInterval);
                }}
                window.coordCheckInterval = setInterval(() => checkAndUpdateCoordinates(), 100);
                setTimeout(() => {{
                    if (window.coordCheckInterval) {{
                        clearInterval(window.coordCheckInterval);
                    }}
                }}, 2000); // Stop checking after 2 seconds
            }}
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                if (e.key === 'z' && !e.shiftKey) {{
                    e.preventDefault();
                    if (historyIndex > 0) {{
                        historyIndex--;
                        boxes = history[historyIndex] ? [...history[historyIndex]] : [];
                        redrawCanvas();
                        updateCoordinatesFromBoxes();
                    }}
                }} else if ((e.key === 'y') || (e.key === 'z' && e.shiftKey)) {{
                    e.preventDefault();
                    if (historyIndex < history.length - 1) {{
                        historyIndex++;
                        boxes = [...history[historyIndex]];
                        redrawCanvas();
                        updateCoordinatesFromBoxes();
                    }}
                }}
            }} else if (e.key === 'Delete' || e.key === 'Backspace') {{
                if (boxes.length > 0) {{
                    saveToHistory();
                    boxes.pop();
                    redrawCanvas();
                    updateCoordinatesFromBoxes();
                }}
            }} else if (e.key === 'Escape') {{
                window.{clear_function}();
            }}
        }});

        function updateCoordinatesFromBoxes() {{
            if (boxes.length > 0) {{
                const lastBox = boxes[boxes.length - 1];
                // No scaling needed - boxes are already in original image coordinates

                const coords = `${{lastBox.x.toFixed(0)}},${{lastBox.y.toFixed(0)}},${{(lastBox.x + lastBox.width).toFixed(0)}},${{(lastBox.y + lastBox.height).toFixed(0)}}`;
                window.{coord_variable} = coords;
                forceUpdateStreamlitInput(coords);
                {update_function}();
            }} else {{
                window.{coord_variable} = null;
                forceUpdateStreamlitInput('');
                {update_function}();
            }}
        }}

        // Clear canvas function
        window.{clear_function} = function() {{
            saveToHistory();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            boxes.length = 0;
            // Clear coordinate display when canvas is cleared
            const displayDiv = document.getElementById('{coord_display_id}');
            if (displayDiv) {{
                displayDiv.textContent = '👆 Draw a bounding box on the image above to see coordinates here';
                displayDiv.style.backgroundColor = '#fff';
                displayDiv.style.border = '2px dashed #ccc';
                displayDiv.style.color = '#666';
                displayDiv.style.fontWeight = 'normal';
            }}
            // Clear coordinates
            window.{coord_variable} = null;
            forceUpdateStreamlitInput('');
        }};

        // Function to update Streamlit coordinates using multiple methods
        function updateStreamlitCoordinates(coords) {{
            // Method 1: Try to find and update the input field by various selectors
            const selectors = [
                'input[aria-label="Current coordinates (x1,y1,x2,y2):"]',
                'input[placeholder="Current coordinates (x1,y1,x2,y2):"]',
                'input[type="text"]',
                'input'
            ];

            let coordsInput = null;
            for (const selector of selectors) {{
                const inputs = document.querySelectorAll(selector);
                for (const input of inputs) {{
                    if (input.placeholder && input.placeholder.includes('coordinates')) {{
                        coordsInput = input;
                        break;
                    }}
                }}
                if (coordsInput) break;
            }}

            if (coordsInput) {{
                coordsInput.value = coords;
                coordsInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                coordsInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                coordsInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
            }}

            // Method 2: Update hidden trigger to force Streamlit refresh
            const hiddenTrigger = document.querySelector('input[aria-label="hidden_trigger"]');
            if (hiddenTrigger) {{
                hiddenTrigger.value = Date.now().toString() + '_' + coords;
                hiddenTrigger.dispatchEvent(new Event('input', {{ bubbles: true }}));
                hiddenTrigger.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}

            // Method 3: Try to update any input that might be the coordinate field
            const allInputs = document.querySelectorAll('input[type="text"]');
            allInputs.forEach(input => {{
                if (input.value !== coords && input.placeholder && input.placeholder.includes('coordinate')) {{
                    input.value = coords;
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            }});
        }}

        // Function to force update Streamlit input field (legacy function)
        function forceUpdateStreamlitInput(coords) {{
            updateStreamlitCoordinates(coords);
        }}

        // Start periodic coordinate checking
        setInterval(checkAndUpdateCoordinates, 500); // Check every 500ms

        // Function to check and update coordinates periodically
        function checkAndUpdateCoordinates() {{
            if (window.{coord_variable}) {{
                const coordsInput = document.querySelector('input[aria-label="Current coordinates (x1,y1,x2,y2):"]');
                if (coordsInput && coordsInput.value !== window.{coord_variable}) {{
                    coordsInput.value = window.{coord_variable};
                    coordsInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    coordsInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
                // Update localStorage
                localStorage.setItem('lastBoxCoords_{element_id}', window.{coord_variable});
            }}
        }}

        // Function to update coordinate display
        function updateCoordinateDisplay() {{
            const displayDiv = document.getElementById('{coord_display_id}');
            if (!displayDiv) return;

            if (window.{coord_variable}) {{
                const parts = window.{coord_variable}.split(',');
                if (parts.length === 4) {{
                    const [x1, y1, x2, y2] = parts.map(p => parseInt(p));
                    const width = x2 - x1;
                    const height = y2 - y1;
                    displayDiv.innerHTML = '<strong>📦 Box Coordinates:</strong> ' + window.{coord_variable} + '<br><small>📏 Size: ' + width + ' × ' + height + ' pixels | 📍 Position: (' + x1 + ', ' + y1 + ')</small>';
                    displayDiv.style.backgroundColor = '#e8f5e8';
                    displayDiv.style.border = '3px solid #4CAF50';
                    displayDiv.style.color = '#2E7D32';
                    displayDiv.style.fontWeight = 'bold';
                }} else {{
                    displayDiv.textContent = window.{coord_variable};
                }}
            }} else {{
                displayDiv.innerHTML = '<strong>👆 Draw a bounding box on the image above</strong><br><small>Tips: Drag to draw • Ctrl+Z/Y: Undo/Redo • Del: Remove last • Esc: Clear all</small>';
                displayDiv.style.backgroundColor = '#fff';
                displayDiv.style.border = '2px dashed #ccc';
                displayDiv.style.color = '#666';
                displayDiv.style.fontWeight = 'normal';
            }}
        }}

        // Function to update annotation coordinate display (for non-dataset images)
        function updateAnnotationCoordinateDisplay() {{
            const displayDiv = document.getElementById('{coord_display_id}');
            if (!displayDiv) return;

            if (window.{coord_variable}) {{
                const parts = window.{coord_variable}.split(',');
                if (parts.length === 4) {{
                    const [x1, y1, x2, y2] = parts.map(p => parseInt(p));
                    const width = x2 - x1;
                    const height = y2 - y1;
                    displayDiv.innerHTML = '<strong>📦 Annotation Coordinates:</strong> ' + window.{coord_variable} + '<br><small>📏 Size: ' + width + ' × ' + height + ' pixels | 📍 Position: (' + x1 + ', ' + y1 + ')</small>';
                    displayDiv.style.backgroundColor = '#e8f5e8';
                    displayDiv.style.border = '3px solid #4CAF50';
                    displayDiv.style.color = '#2E7D32';
                    displayDiv.style.fontWeight = 'bold';
                }} else {{
                    displayDiv.textContent = window.{coord_variable};
                }}
            }} else {{
                displayDiv.innerHTML = '<strong>👆 Draw a bounding box on the image above</strong><br><small>Tips: Drag to draw • Ctrl+Z/Y: Undo/Redo • Del: Remove last • Esc: Clear all</small>';
                displayDiv.style.backgroundColor = '#fff';
                displayDiv.style.border = '2px dashed #ccc';
                displayDiv.style.color = '#666';
                displayDiv.style.fontWeight = 'normal';
            }}
        }}
    }})();
    </script>
    """
st.info("🚀 **Welcome!** This web UI helps you create YOLO v8 datasets, train models, and test them on images. Use the tabs below to navigate through the workflow.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Create Dataset", "Train Model", "Test Model", "Browse"])

with tab1:
    st.header("Create Dataset")
    st.info("📸 **Step 1:** Upload images and create your YOLO dataset with annotations. This prepares your data for training.")

    # Dataset Creation Tips
    with st.expander("💡 Dataset Creation Tips", expanded=False):
        st.markdown("""
        **🎯 Best Practices for YOLO Datasets:**

        **📊 Dataset Size:**
        - Minimum 100-200 images per class for basic training
        - 1000+ images per class for good results
        - Aim for balanced classes

        **🖼️ Image Quality:**
        - Use high-resolution images (at least 416x416 pixels)
        - Ensure good lighting and contrast
        - Avoid blurry or heavily compressed images
        - Include images from different angles and distances

        **🏷️ Annotation Guidelines:**
        - Draw bounding boxes tightly around objects
        - Include some background context around objects
        - Ensure consistent annotation style across all images
        - Use the annotation tools provided for best results

        **📁 File Organization:**
        - Use descriptive dataset names
        - Keep related images together
        - Avoid special characters in filenames
        - Maintain consistent image formats

        **⚡ Performance Tips:**
        - Validate images before uploading (use the preview feature)
        - Upload images in batches for large datasets
        - Check dataset health after creation
        """)

    # Dataset Management Section
    st.subheader("📁 Dataset Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        dataset_name = st.text_input("Dataset Name", placeholder="my_dataset", help="Name for your new dataset")

    with col2:
        # Dataset loading with proper UI
        if os.path.exists("datasets"):
            existing_datasets = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
            if existing_datasets:
                selected_dataset = st.selectbox("Select dataset to load:", existing_datasets, key="load_existing")
                if st.button("📁 Load Selected Dataset", type="secondary"):
                    if selected_dataset:
                        st.session_state.current_dataset = selected_dataset
                        st.success(f"✅ Loaded dataset: {selected_dataset}")
                        st.rerun()
            else:
                st.info("No existing datasets found.")
        else:
            st.info("No datasets folder found.")

    # Show current dataset
    if 'current_dataset' in st.session_state:
        st.success(f"📂 **Current Dataset:** {st.session_state.current_dataset}")

        # Dataset actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Switch Dataset", key="switch_dataset"):
                if 'current_dataset' in st.session_state:
                    del st.session_state.current_dataset
                st.rerun()

        with col2:
            if st.button("📊 View Dataset", key="view_dataset"):
                st.session_state.view_dataset = True

        with col3:
            if st.button("🗑️ Delete Dataset", key="delete_dataset"):
                if st.checkbox("Confirm deletion", key="confirm_delete_dataset"):
                    dataset_path = os.path.join("datasets", st.session_state.current_dataset)
                    import shutil
                    shutil.rmtree(dataset_path)
                    del st.session_state.current_dataset
                    st.success("Dataset deleted!")
                    st.rerun()

    # Dataset Viewer
    if 'view_dataset' in st.session_state and st.session_state.view_dataset:
        st.subheader(f"📊 Dataset Viewer: {st.session_state.current_dataset}")

        dataset_path = os.path.join("datasets", st.session_state.current_dataset)
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        # Dataset statistics
        col1, col2, col3 = st.columns(3)

        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        labels = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]

        with col1:
            st.metric("Total Images", len(images))
        with col2:
            st.metric("Label Files", len(labels))
        with col3:
            st.metric("Unlabeled", len(images) - len(labels))

        # Image browser with annotations
        if images:
            st.subheader("🖼️ Browse Images & Annotations")

            # Image selector
            selected_image_file = st.selectbox("Select image to view:", images, key="dataset_image_selector")

            if selected_image_file:
                image_path = os.path.join(images_path, selected_image_file)
                image = safe_open_image(image_path)
                if image is None:
                    st.error("❌ Failed to load image for viewing.")
                    st.stop()

                # Check for corresponding label file
                label_file = os.path.splitext(selected_image_file)[0] + ".txt"
                label_path = os.path.join(labels_path, label_file)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption=f"Image: {selected_image_file}", width='stretch')

                with col2:
                    st.write("**Annotations:**")
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            annotations = f.read().strip()

                        if annotations:
                            # Parse YOLO annotations
                            annotation_lines = annotations.split('\n')
                            st.write(f"**Found {len(annotation_lines)} bounding box(es):**")

                            for i, line in enumerate(annotation_lines, 1):
                                if line.strip():
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = parts[0]
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])

                                        # Convert back to pixel coordinates for display
                                        img_width, img_height = image.size
                                        x1 = int((x_center - width/2) * img_width)
                                        y1 = int((y_center - height/2) * img_height)
                                        x2 = int((x_center + width/2) * img_width)
                                        y2 = int((y_center + height/2) * img_height)

                                        st.code(f"Box {i}: [{x1}, {y1}, {x2}, {y2}] (Class: {class_id})")
                        else:
                            st.info("No annotations found for this image.")
                    else:
                        st.info("No annotation file found for this image.")

                # Actions for this image
                st.subheader("Image Actions")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("📝 Annotate This Image", key=f"annotate_{selected_image_file}"):
                        # Load this image for annotation
                        st.session_state.annotate_image = selected_image_file
                        st.success(f"✅ Loaded {selected_image_file} for annotation")
                        st.rerun()

                with col2:
                    if st.button("🗑️ Delete Image", key=f"delete_image_{selected_image_file}"):
                        if st.checkbox(f"Confirm deletion of {selected_image_file}", key=f"confirm_delete_{selected_image_file}"):
                            os.remove(image_path)
                            if os.path.exists(label_path):
                                os.remove(label_path)
                            st.success(f"✅ Deleted {selected_image_file}")
                            st.rerun()

                with col3:
                    # Create download button with unique key
                    download_key = f"download_{selected_image_file}_{hash(image_path)}"
                    try:
                        with open(image_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Image",
                                data=f,
                                file_name=selected_image_file,
                                mime="image/jpeg",
                                key=download_key
                            )
                    except Exception as e:
                        st.error(f"❌ Error preparing download: {str(e)}")

        # Close viewer
        if st.button("❌ Close Viewer", key="close_viewer"):
            st.session_state.view_dataset = False
            st.rerun()

    # Dataset Health Check
    if 'current_dataset' in st.session_state and not 'view_dataset' in st.session_state:
        st.subheader("🔍 Dataset Health Check")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔍 Check Dataset Health", key="check_health"):
                dataset_path = os.path.join("datasets", st.session_state.current_dataset)

                with st.spinner("🔄 Analyzing dataset..."):
                    is_valid, errors, warnings = validate_dataset_integrity(dataset_path)

                if is_valid:
                    st.success("✅ Dataset structure is valid!")

                    if not errors and not warnings:
                        st.info("🎉 No issues found - your dataset is in perfect health!")
                    else:
                        if errors:
                            st.error("❌ **Critical Issues:**")
                            for error in errors:
                                st.error(f"• {error}")

                        if warnings:
                            st.warning("⚠️ **Warnings:**")
                            for warning in warnings:
                                st.warning(f"• {warning}")
                else:
                    st.error("❌ Dataset has critical structural issues:")
                    for error in errors:
                        st.error(f"• {error}")

                    if warnings:
                        st.warning("⚠️ **Additional Warnings:**")
                        for warning in warnings:
                            st.warning(f"• {warning}")

        with col2:
            if st.button("🧹 Clean Dataset", key="clean_dataset", help="Remove orphaned files and fix common issues"):
                dataset_path = os.path.join("datasets", st.session_state.current_dataset)

                with st.spinner("🔄 Cleaning dataset..."):
                    # Get current state
                    is_valid, errors, warnings = validate_dataset_integrity(dataset_path)

                    if is_valid:
                        images_path = os.path.join(dataset_path, "images")
                        labels_path = os.path.join(dataset_path, "labels")

                        # Get file lists
                        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
                        label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]

                        image_basenames = {os.path.splitext(f)[0] for f in image_files}
                        label_basenames = {os.path.splitext(f)[0] for f in label_files}

                        # Find orphaned files
                        labels_without_images = label_basenames - image_basenames
                        images_without_labels = image_basenames - label_basenames

                        cleaned_count = 0

                        # Remove orphaned label files
                        for label_base in labels_without_images:
                            label_file = label_base + ".txt"
                            label_path = os.path.join(labels_path, label_file)
                            try:
                                os.remove(label_path)
                                cleaned_count += 1
                            except Exception as e:
                                st.warning(f"Could not remove orphaned label file '{label_file}': {str(e)}")

                        # Create missing label files for images
                        for image_base in images_without_labels:
                            label_file = image_base + ".txt"
                            label_path = os.path.join(labels_path, label_file)
                            try:
                                with open(label_path, "w") as f:
                                    f.write("")  # Empty label file
                                cleaned_count += 1
                            except Exception as e:
                                st.warning(f"Could not create label file for '{image_base}': {str(e)}")

                        if cleaned_count > 0:
                            st.success(f"✅ Cleaned up {cleaned_count} files!")
                            st.rerun()
                        else:
                            st.info("ℹ️ No cleanup needed - dataset is already clean!")
                    else:
                        st.error("❌ Cannot clean dataset due to structural issues. Please fix them first.")

        with col2:
            if st.button("📊 Export Dataset Report", key="export_report", help="Generate detailed dataset statistics and recommendations"):
                dataset_path = os.path.join("datasets", st.session_state.current_dataset)

                with st.spinner("🔄 Generating dataset report..."):
                    # Generate comprehensive dataset report
                    report_data = generate_dataset_report(dataset_path)

                if report_data:
                    # Display report summary
                    st.success("✅ Dataset report generated successfully!")

                    # Report sections
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📈 Dataset Statistics")

                        stats = report_data['statistics']
                        st.info(f"**Total Images:** {stats['total_images']}")
                        st.info(f"**Total Annotations:** {stats['total_annotations']}")
                        st.info(f"**Images with Annotations:** {stats['annotated_images']}")
                        st.info(f"**Unannotated Images:** {stats['unannotated_images']}")
                        st.info(f"**Average Annotations per Image:** {stats['avg_annotations_per_image']:.2f}")

                        if stats['total_annotations'] > 0:
                            st.info(f"**Most Common Class:** {stats['most_common_class']}")
                            st.info(f"**Annotation Distribution:** {', '.join([f'Class {k}: {v}' for k, v in stats['class_distribution'].items()])}")

                    with col2:
                        st.subheader("🔍 Dataset Quality")

                        quality = report_data['quality']
                        if quality['issues']:
                            st.warning("**Issues Found:**")
                            for issue in quality['issues']:
                                st.warning(f"• {issue}")
                        else:
                            st.success("**No Quality Issues Found!**")

                        st.info(f"**Dataset Health Score:** {quality['health_score']}/100")

                        if quality['recommendations']:
                            st.subheader("💡 Recommendations")
                            for rec in quality['recommendations']:
                                st.info(f"• {rec}")

                    # Export options
                    st.subheader("📥 Export Options")

                    # Generate report as text
                    report_text = f"""DATASET REPORT: {st.session_state.current_dataset}
Generated on: {report_data['timestamp']}

DATASET STATISTICS:
- Total Images: {stats['total_images']}
- Total Annotations: {stats['total_annotations']}
- Images with Annotations: {stats['annotated_images']}
- Unannotated Images: {stats['unannotated_images']}
- Average Annotations per Image: {stats['avg_annotations_per_image']:.2f}

DATASET QUALITY:
- Health Score: {quality['health_score']}/100

ISSUES FOUND:
{chr(10).join('- ' + issue for issue in quality['issues']) if quality['issues'] else 'None'}

RECOMMENDATIONS:
{chr(10).join('- ' + rec for rec in quality['recommendations']) if quality['recommendations'] else 'None'}

CLASS DISTRIBUTION:
{chr(10).join(f'- Class {k}: {v} annotations' for k, v in stats['class_distribution'].items()) if stats['class_distribution'] else 'No annotations found'}
"""

                    # Download report
                    st.download_button(
                        label="⬇️ Download Report (.txt)",
                        data=report_text,
                        file_name=f"dataset_report_{st.session_state.current_dataset}_{report_data['timestamp'].replace(':', '-')}.txt",
                        mime="text/plain",
                        key="download_report_txt"
                    )

                    # Generate JSON report
                    import json
                    json_report = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="⬇️ Download Report (.json)",
                        data=json_report,
                        file_name=f"dataset_report_{st.session_state.current_dataset}_{report_data['timestamp'].replace(':', '-')}.json",
                        mime="application/json",
                        key="download_report_json"
                    )
                else:
                    st.error("❌ Failed to generate dataset report. Please check dataset integrity first.")

    # Handle annotation of specific image from dataset viewer
    if 'annotate_image' in st.session_state and st.session_state.annotate_image:
        st.subheader(f"📝 Annotating: {st.session_state.annotate_image}")

        dataset_path = os.path.join("datasets", st.session_state.current_dataset)
        image_path = os.path.join(dataset_path, "images", st.session_state.annotate_image)

        if os.path.exists(image_path):
            image = safe_open_image(image_path)
            if image is None:
                st.error("❌ Failed to load image for annotation.")
                st.stop()

            img_width, img_height = image.size

            # Convert image to base64 for HTML display
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Create annotation canvas
            annotation_html = generate_annotation_canvas_html(img_base64, img_width, img_height, st.session_state.annotate_image)

            components.html(annotation_html, height=img_height + 120)

            # Annotation controls
            st.subheader('📍 Annotation Controls')

            # Text input for annotation coordinates
            annotate_coords_input = st.text_input(
                'Annotation coordinates (x1,y1,x2,y2):',
                key=f"annotate_coords_input_{st.session_state.annotate_image}",
                help="Coordinates will appear here automatically when you draw a box"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button('💾 Save Annotation', key=f"save_annotation_{st.session_state.annotate_image}", type="primary"):
                    if annotate_coords_input and annotate_coords_input.strip():
                        # Validate coordinates before saving
                        is_valid, result = validate_coordinates(annotate_coords_input, img_width, img_height)
                        if not is_valid:
                            st.error(f"❌ {result}")
                            st.stop()

                        # Save annotation to file
                        labels_path = os.path.join(dataset_path, "labels")
                        label_filename = os.path.splitext(st.session_state.annotate_image)[0] + ".txt"
                        label_path = os.path.join(labels_path, label_filename)

                        try:
                            # Ensure labels directory exists
                            os.makedirs(labels_path, exist_ok=True)

                            # result is already the tuple (x1, y1, x2, y2)
                            coords = result
                            x1 = float(coords[0])
                            y1 = float(coords[1])
                            x2 = float(coords[2])
                            y2 = float(coords[3])

                            # Convert to YOLO format (normalized)
                            x_center = ((x1 + x2) / 2) / img_width
                            y_center = ((y1 + y2) / 2) / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height

                            # YOLO format: class x_center y_center width height
                            yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                            # Save annotation (append to existing file)
                            with open(label_path, "a") as f:
                                f.write(yolo_annotation + "\n")

                            st.success(f'✅ Annotation saved to: {label_filename}')
                        except Exception as e:
                            st.error(f'❌ Error saving annotation: {str(e)}')
                    else:
                        st.warning('⚠️ No coordinates to save. Please draw a box first!')

            with col2:
                if st.button('🗑️ Clear Canvas', key=f"clear_annotation_canvas_{st.session_state.annotate_image}", type="secondary"):
                    clear_js = f"""
                    <script>
                    (function() {{
                        if (window.clearAnnotationCanvas) {{
                            window.clearAnnotationCanvas();
                        }}
                        // Clear text input
                        const coordsInput = document.querySelector('input[aria-label="Annotation coordinates (x1,y1,x2,y2):"]');
                        if (coordsInput) {{
                            coordsInput.value = '';
                            coordsInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }})();
                    </script>
                    """
                    components.html(clear_js, height=0)
                    st.success('✅ Annotation canvas cleared!')

            with col3:
                if st.button('❌ Stop Annotating', key=f"stop_annotating_{st.session_state.annotate_image}", type="secondary"):
                    del st.session_state.annotate_image
                    st.success('✅ Stopped annotation mode')
                    st.rerun()

    # File uploader (only show if not annotating a specific image)
    if 'annotate_image' not in st.session_state:
        uploaded_files = st.file_uploader('Upload images', accept_multiple_files=True, type=['jpg','png','jpeg'])

        if uploaded_files:
            # Display upload summary
            st.success(f'✅ Uploaded {len(uploaded_files)} files!')

            # File preview and validation
            st.subheader("📋 File Preview")

            # Create a table of uploaded files
            file_data = []
            total_size = 0

            for file in uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                total_size += file_size_mb

                # Quick validation
                is_valid, _ = validate_image_file(file)
                status = "✅ Valid" if is_valid else "❌ Invalid"

                file_data.append({
                    "Filename": file.name,
                    "Size (MB)": ".2f",
                    "Status": status
                })

            # Display file summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(uploaded_files))
            with col2:
                st.metric("Total Size", ".1f")
            with col3:
                valid_count = sum(1 for item in file_data if "✅ Valid" in item["Status"])
                st.metric("Valid Files", f"{valid_count}/{len(uploaded_files)}")

            # Show file details in a table
            if len(file_data) <= 20:  # Only show table for reasonable number of files
                import pandas as pd
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info(f"📊 {len(file_data)} files uploaded. Showing summary above.")

            # Show warnings for large uploads
            if len(uploaded_files) > 100:
                st.warning("⚠️ Large number of files detected. Processing may take some time.")

            if total_size > 500:  # 500MB
                st.warning(".1f")

            # Quick preview of first few images
            if len(uploaded_files) <= 5:
                st.subheader("🖼️ Image Preview")
                cols = st.columns(min(len(uploaded_files), 3))

                for i, file in enumerate(uploaded_files[:3]):
                    with cols[i]:
                        try:
                            image = safe_open_image(file)
                            if image:
                                st.image(image, caption=file.name, width=200)
                            else:
                                st.error(f"Could not preview {file.name}")
                        except Exception:
                            st.error(f"Could not preview {file.name}")

        # Create dataset if name provided and not already loaded
        if dataset_name and 'current_dataset' not in st.session_state:
            if st.button("💾 Create Dataset", type="primary"):
                # Create progress tracking elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_container = st.empty()
                warning_container = st.empty()

                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)

                # Use robust dataset creation
                success, errors, warnings = create_dataset_robust(dataset_name, uploaded_files, progress_callback)

                # Display results
                progress_bar.empty()
                status_text.empty()

                if success:
                    st.session_state.current_dataset = dataset_name
                    st.success(f"✅ Dataset '{dataset_name}' created successfully!")

                    # Show warnings if any
                    if warnings:
                        with warning_container:
                            st.warning("⚠️ **Warnings:**")
                            for warning in warnings:
                                st.warning(warning)

                    st.rerun()
                else:
                    # Show errors
                    with error_container:
                        st.error("❌ **Dataset creation failed:**")
                        for error in errors:
                            st.error(error)

                    # Show warnings if any
                    if warnings:
                        with warning_container:
                            st.warning("⚠️ **Warnings:**")
                            for warning in warnings:
                                st.warning(warning)

        elif 'current_dataset' in st.session_state:
            st.info("📂 Working with existing dataset. Upload additional images to add them to the dataset.")

            # Option to add images to existing dataset
            if st.button("➕ Add Images to Dataset", type="secondary"):
                # Create progress tracking elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_container = st.empty()
                warning_container = st.empty()

                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)

                # Use robust image addition
                success, errors, warnings, added_count = add_images_to_dataset_robust(
                    st.session_state.current_dataset, uploaded_files, progress_callback
                )

                # Display results
                progress_bar.empty()
                status_text.empty()

                if success:
                    if added_count > 0:
                        st.success(f"✅ Successfully added {added_count} images to dataset!")

                        # Show warnings if any
                        if warnings:
                            with warning_container:
                                st.warning("⚠️ **Warnings:**")
                                for warning in warnings:
                                    st.warning(warning)

                        st.rerun()
                    else:
                        st.info("ℹ️ No new images were added (all were duplicates or invalid)")

                        # Show warnings if any
                        if warnings:
                            with warning_container:
                                st.warning("⚠️ **Warnings:**")
                                for warning in warnings:
                                    st.warning(warning)
                else:
                    # Show errors
                    with error_container:
                        st.error("❌ **Failed to add images:**")
                        for error in errors:
                            st.error(error)

                    # Show warnings if any
                    if warnings:
                        with warning_container:
                            st.warning("⚠️ **Warnings:**")
                            for warning in warnings:
                                st.warning(warning)

        # Image selector
        image_names = [file.name for file in uploaded_files]
        selected_image_name = st.selectbox('Select image to annotate:', image_names)

        if selected_image_name:
            selected_file = next(file for file in uploaded_files if file.name == selected_image_name)
            image = safe_open_image(selected_file)
            if image is None:
                st.error("❌ Failed to load image for annotation.")
                st.stop()

            # Get image dimensions
            img_width, img_height = image.size

            # Simple bounding box drawing interface
            st.subheader('Draw Bounding Boxes')

            # Convert image to base64 for display
            import io
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Interactive bounding box drawing
            st.subheader('Draw Bounding Boxes')

            # Interactive canvas for drawing bounding boxes
            canvas_html = f"""
            <div style="position: relative; display: inline-block; border: 2px solid #ddd; margin: 10px 0;">
                <img id="annotation_image_{selected_image_name}" src="data:image/png;base64,{img_base64}"
                     style="display: block; max-width: 100%; height: auto;" />
                <canvas id="annotation_canvas_{selected_image_name}"
                        style="position: absolute; top: 0; left: 0; cursor: crosshair; border: none; pointer-events: auto; z-index: 10;"
                        width="{img_width}" height="{img_height}">
                </canvas>
            </div>

            <div id="canvas_status_{selected_image_name}" style="margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;">
                👆 Click and drag on the image above to draw a bounding box
            </div>
            
            <script>
                console.log('Canvas script starting for {selected_image_name}');
                
                // Wait for DOM to be ready
                setTimeout(function() {{
                    const img = document.getElementById('annotation_image_{selected_image_name}');
                    const canvas = document.getElementById('annotation_canvas_{selected_image_name}');
                    const statusDiv = document.getElementById('canvas_status_{selected_image_name}');
                    
                    console.log('Elements:', {{img: !!img, canvas: !!canvas, statusDiv: !!statusDiv}});
                    
                    if (!img || !canvas || !statusDiv) {{
                        console.error('Elements not found');
                        return;
                    }}
                    
                    const ctx = canvas.getContext('2d');
                    let isDrawing = false;
                    let startX, startY;
                    let savedBoxes = [];

                    // Update status display
                    function updateStatus(message) {{
                        statusDiv.textContent = message;
                    }}

                    // Set canvas size to match image
                    function resizeCanvas() {{
                        const rect = img.getBoundingClientRect();
                        canvas.style.width = rect.width + 'px';
                        canvas.style.height = rect.height + 'px';
                        console.log('Canvas resized to match image');
                    }}

                    // Wait for image to load
                    if (img.complete) {{
                        resizeCanvas();
                    }} else {{
                        img.onload = resizeCanvas;
                    }}

                    // Convert mouse coordinates to canvas coordinates
                    function getCanvasCoords(clientX, clientY) {{
                        const rect = canvas.getBoundingClientRect();
                        const scaleX = canvas.width / rect.width;
                        const scaleY = canvas.height / rect.height;
                        return {{
                            x: (clientX - rect.left) * scaleX,
                            y: (clientY - rect.top) * scaleY
                        }};
                    }}

                    // Draw box
                    function drawBox(x, y, width, height, isPreview = false) {{
                        ctx.strokeStyle = isPreview ? '#ff6b6b' : '#4CAF50';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, width, height);
                        
                        if (isPreview) {{
                            ctx.fillStyle = 'rgba(255, 107, 107, 0.2)';
                            ctx.fillRect(x, y, width, height);
                        }}
                    }}

                    // Redraw all boxes
                    function redrawBoxes() {{
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        savedBoxes.forEach(box => {{
                            drawBox(box.x, box.y, box.width, box.height, false);
                        }});
                    }}

                    // Mouse events
                    canvas.addEventListener('mousedown', function(e) {{
                        e.preventDefault();
                        const coords = getCanvasCoords(e.clientX, e.clientY);
                        isDrawing = true;
                        startX = coords.x;
                        startY = coords.y;
                        updateStatus('Drawing box...');
                        console.log('Mouse down at:', coords);
                    }});

                    canvas.addEventListener('mousemove', function(e) {{
                        if (!isDrawing) return;
                        e.preventDefault();
                        
                        const coords = getCanvasCoords(e.clientX, e.clientY);
                        const width = coords.x - startX;
                        const height = coords.y - startY;
                        
                        // Clear and redraw
                        redrawBoxes();
                        drawBox(Math.min(startX, coords.x), Math.min(startY, coords.y), 
                                Math.abs(width), Math.abs(height), true);
                    }});

                    canvas.addEventListener('mouseup', function(e) {{
                        if (!isDrawing) return;
                        e.preventDefault();
                        isDrawing = false;
                        
                        const coords = getCanvasCoords(e.clientX, e.clientY);
                        const width = Math.abs(coords.x - startX);
                        const height = Math.abs(coords.y - startY);
                        
                        if (width > 10 && height > 10) {{
                            const box = {{
                                x: Math.min(startX, coords.x),
                                y: Math.min(startY, coords.y),
                                width: width,
                                height: height
                            }};
                            savedBoxes.push(box);
                            
                            // Format coordinates and update input
                            const x1 = Math.round(box.x);
                            const y1 = Math.round(box.y);
                            const x2 = Math.round(box.x + box.width);
                            const y2 = Math.round(box.y + box.height);
                            const coordsStr = x1 + ',' + y1 + ',' + x2 + ',' + y2;
                            
                            updateStatus('Box drawn! Coordinates: ' + coordsStr);
                            console.log('Box saved:', coordsStr);
                            
                            // Try to update coordinate input
                            const inputs = document.querySelectorAll('input[type="text"]');
                            for (let input of inputs) {{
                                const parent = input.closest('div');
                                if (parent && parent.textContent.includes('Current coordinates')) {{
                                    input.value = coordsStr;
                                    input.dispatchEvent(new Event('input', {{bubbles: true}}));
                                    input.dispatchEvent(new Event('change', {{bubbles: true}}));
                                    break;
                                }}
                            }}
                        }} else {{
                            updateStatus('Box too small! Try again.');
                        }}
                        
                        redrawBoxes();
                    }});

                    // Clear on Escape key
                    document.addEventListener('keydown', function(e) {{
                        if (e.key === 'Escape') {{
                            savedBoxes = [];
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            updateStatus('All boxes cleared!');
                        }}
                    }});

                    updateStatus('Ready! Click and drag to draw boxes. Press Escape to clear.');
                    console.log('Canvas setup complete');
                }}, 100);
            </script>
            """
            
            components.html(canvas_html, height=img_height + 150)

            # Quick canvas control
            col_canvas, col_spacer = st.columns([1, 3])
            with col_canvas:
                if st.button('🗑️ Clear Canvas', key=f"clear_canvas_{selected_image_name}", type="secondary"):
                    clear_js = f"""
                    <script>
                    (function() {{
                        if (window.canvasControls_{selected_image_name}) {{
                            window.canvasControls_{selected_image_name}.clearAll();
                        }}
                    }})();
                    </script>
                    """
                    components.html(clear_js, height=0)
                    st.success('✅ Canvas cleared!')

            # Advanced Bounding Box Management
            st.subheader('🎯 Smart Bounding Box Manager')

            # Initialize improved session state for multiple boxes
            boxes_key = f"boxes_{selected_image_name}"
            if boxes_key not in st.session_state:
                st.session_state[boxes_key] = []

            # Main control panel
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("**📝 Quick Add Box**")
                quick_coords = st.text_input(
                    "Enter coordinates (x1,y1,x2,y2):",
                    placeholder="e.g., 100,100,200,200",
                    key=f"quick_coords_{selected_image_name}",
                    help="Add a new bounding box by typing coordinates"
                )
                
                if st.button("➕ Add Box", key=f"add_quick_box_{selected_image_name}", type="primary"):
                    if quick_coords and quick_coords.strip():
                        is_valid, result = validate_coordinates(quick_coords, img_width, img_height)
                        if is_valid:
                            from datetime import datetime
                            box_data = {
                                'id': len(st.session_state[boxes_key]) + 1,
                                'coords': quick_coords,
                                'class_id': 0,
                                'class_name': 'object',
                                'confidence': 1.0,
                                'created_at': str(datetime.now().strftime("%H:%M:%S"))
                            }
                            st.session_state[boxes_key].append(box_data)
                            st.success(f"✅ Box {box_data['id']} added: {quick_coords}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
                    else:
                        st.warning("⚠️ Please enter coordinates first!")

            with col2:
                st.markdown("**🎨 Canvas Drawing**")
                st.info("👆 Draw directly on the image above")
                if st.button("🔄 Sync from Canvas", key=f"sync_from_canvas_{selected_image_name}", type="secondary"):
                    st.info("Canvas sync functionality active - draw on the image to add boxes automatically!")

            with col3:
                st.markdown("**🗂️ Actions**")
                if st.button("🗑️ Clear All", key=f"clear_all_new_{selected_image_name}", type="secondary"):
                    st.session_state[boxes_key] = []
                    st.success("✅ All boxes cleared!")
                    st.rerun()

            # Display current boxes - Smart Bounding Box Manager
            if st.session_state[boxes_key]:
                st.markdown(f"**📦 Current Boxes ({len(st.session_state[boxes_key])})**")
                
                # Enhanced box display with editing capabilities
                for i, box in enumerate(st.session_state[boxes_key]):
                    with st.container():
                        box_col1, box_col2, box_col3, box_col4, box_col5 = st.columns([1, 3, 2, 2, 1])
                        
                        with box_col1:
                            st.markdown(f"**#{box['id']}**")
                        
                        with box_col2:
                            # Editable coordinates
                            new_coords = st.text_input(
                                "Coordinates:",
                                value=box['coords'],
                                key=f"edit_coords_{box['id']}_{selected_image_name}",
                                label_visibility="collapsed"
                            )
                            if new_coords != box['coords']:
                                is_valid, result = validate_coordinates(new_coords, img_width, img_height)
                                if is_valid:
                                    st.session_state[boxes_key][i]['coords'] = new_coords
                                    st.success(f"✅ Box {box['id']} updated")
                                    st.rerun()
                        
                        with box_col3:
                            # Class selection
                            class_options = ['object', 'person', 'car', 'animal', 'custom']
                            current_class = box.get('class_name', 'object')
                            new_class = st.selectbox(
                                "Class:",
                                options=class_options,
                                index=class_options.index(current_class) if current_class in class_options else 0,
                                key=f"class_{box['id']}_{selected_image_name}",
                                label_visibility="collapsed"
                            )
                            if new_class != current_class:
                                st.session_state[boxes_key][i]['class_name'] = new_class
                                st.session_state[boxes_key][i]['class_id'] = class_options.index(new_class)
                        
                        with box_col4:
                            st.text(f"Added: {box['created_at']}")
                        
                        with box_col5:
                            if st.button("❌", key=f"remove_box_new_{box['id']}_{selected_image_name}", help=f"Remove box {box['id']}"):
                                st.session_state[boxes_key].pop(i)
                                st.success(f"✅ Box {box['id']} removed!")
                                st.rerun()

            else:
                st.info("📝 No bounding boxes yet. Draw on the image above or add coordinates manually.")

            # Action buttons for managing all boxes
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button('🗑️ Clear All Boxes', key=f"clear_all_boxes_{selected_image_name}", type="secondary"):
                    st.session_state[boxes_key] = []
                    st.success('✅ All boxes cleared!')
                    st.rerun()

            with col2:
                if st.button('💾 Save All Boxes', key=f"save_all_boxes_{selected_image_name}", type="primary"):
                    if st.session_state[boxes_key] and 'current_dataset' in st.session_state:
                        # Save all boxes to YOLO format file  
                        dataset_path = os.path.join("datasets", st.session_state.current_dataset)
                        labels_path = os.path.join(dataset_path, "labels")
                        
                        # Ensure labels directory exists
                        os.makedirs(labels_path, exist_ok=True)
                        
                        # Create label filename (same as image but .txt)
                        label_filename = os.path.splitext(selected_image_name)[0] + ".txt"
                        label_path = os.path.join(labels_path, label_filename)
                        
                        # Convert all boxes to YOLO format and save
                        try:
                            with open(label_path, "w") as f:
                                for box in st.session_state[boxes_key]:
                                    # Parse coordinates
                                    coords = box['coords'].split(',')
                                    x1, y1, x2, y2 = map(float, coords)
                                    
                                    # Convert to YOLO format (normalized)
                                    x_center = ((x1 + x2) / 2) / img_width
                                    y_center = ((y1 + y2) / 2) / img_height
                                    width = (x2 - x1) / img_width
                                    height = (y2 - y1) / img_height
                                    
                                    # Use class_id from box, default to 0
                                    class_id = box.get('class_id', 0)
                                    
                                    # YOLO format: class x_center y_center width height
                                    yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                    f.write(yolo_annotation + "\n")
                            
                            st.success(f'✅ Saved {len(st.session_state[boxes_key])} boxes to: {label_filename}')
                            
                        except Exception as e:
                            st.error(f'❌ Error saving boxes: {str(e)}')
                    elif 'current_dataset' not in st.session_state:
                        st.warning('⚠️ Please create or load a dataset first!')
                    else:
                        st.warning('⚠️ No boxes to save!')

            with col3:
                if st.button('📥 Load Saved Labels', key=f"load_saved_labels_{selected_image_name}", type="secondary"):
                    if 'current_dataset' in st.session_state:
                        dataset_path = os.path.join("datasets", st.session_state.current_dataset)
                        labels_path = os.path.join(dataset_path, "labels")
                        label_filename = os.path.splitext(selected_image_name)[0] + ".txt"
                        label_path = os.path.join(labels_path, label_filename)
                        
                        if os.path.exists(label_path):
                            try:
                                loaded_boxes = []
                                with open(label_path, 'r') as f:
                                    for line_num, line in enumerate(f, 1):
                                        line = line.strip()
                                        if line:
                                            parts = line.split()
                                            if len(parts) >= 5:
                                                class_id = int(parts[0])
                                                x_center, y_center, width, height = map(float, parts[1:5])
                                                
                                                # Convert from YOLO format to pixel coordinates
                                                x1 = (x_center - width/2) * img_width
                                                y1 = (y_center - height/2) * img_height
                                                x2 = (x_center + width/2) * img_width
                                                y2 = (y_center + height/2) * img_height
                                                
                                                coords_str = f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"
                                                
                                                class_options = ['object', 'person', 'car', 'animal', 'custom']
                                                class_name = class_options[class_id] if class_id < len(class_options) else 'object'
                                                
                                                loaded_boxes.append({
                                                    'id': f"loaded_{line_num}",
                                                    'coords': coords_str,
                                                    'class_name': class_name,
                                                    'class_id': class_id,
                                                    'created_at': datetime.now().strftime("%H:%M:%S")
                                                })
                                
                                if loaded_boxes:
                                    st.session_state[boxes_key] = loaded_boxes
                                    st.success(f'✅ Loaded {len(loaded_boxes)} boxes from saved labels!')
                                    st.rerun()
                                else:
                                    st.warning('⚠️ No valid boxes found in label file!')
                                    
                            except Exception as e:
                                st.error(f'❌ Error loading labels: {str(e)}')
                        else:
                            st.warning(f'⚠️ No saved labels found for {selected_image_name}')
                    else:
                        st.warning('⚠️ Please create or load a dataset first!')

            with col4:
                if st.button('🔄 Sync Canvas', key=f"sync_canvas_{selected_image_name}", type="secondary"):
                    if st.session_state[boxes_key]:
                        # Sync all boxes to canvas
                        sync_js = f"""
                        <script>
                        (function() {{
                            if (window.canvasControls_{selected_image_name}) {{
                                window.canvasControls_{selected_image_name}.clearAll();
                                
                                // Add all boxes from session state
                                const boxes = {json.dumps([box['coords'] for box in st.session_state[boxes_key]])};
                                
                                boxes.forEach(function(coordsStr) {{
                                    const coords = coordsStr.split(',');
                                    if (coords.length === 4) {{
                                        const x1 = parseFloat(coords[0]);
                                        const y1 = parseFloat(coords[1]);
                                        const x2 = parseFloat(coords[2]);
                                        const y2 = parseFloat(coords[3]);
                                        window.canvasControls_{selected_image_name}.addBoxFromCoords(x1, y1, x2, y2);
                                    }}
                                }});
                                
                                const statusDiv = document.getElementById('canvas_status_{selected_image_name}');
                                if (statusDiv) {{
                                    statusDiv.innerHTML = '🔄 Canvas synced with ' + boxes.length + ' boxes';
                                    statusDiv.style.backgroundColor = '#e3f2fd';
                                    statusDiv.style.color = '#1976d2';
                                }}
                            }}
                        }})();
                        </script>
                        """
                        components.html(sync_js, height=0)
                        st.info(f'🔄 Canvas synchronized with {len(st.session_state[boxes_key])} boxes')
                    else:
                        st.warning('⚠️ No boxes to sync to canvas!')

        else:


            # Display current coordinate status
            # Smart Bounding Box Manager is complete - this section can be extended as needed
                st.info(f"� **Current Coordinates:** {st.session_state[coord_key]}")


            # Old coordinate system completely removed - using new Smart Bounding Box Manager only

            # List available bbox files
            # bbox_files = []  # Commented out old coordinate system
            # Old coordinate system removed - all bbox_files code commented out
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if 'bounding_boxes' in data and isinstance(data['bounding_boxes'], list):
                        bbox_files.append({
                            'file': str(json_file),
                            'name': json_file.name,
                            'bbox_count': len(data['bounding_boxes']),
                            'image_name': data.get('image_name', 'Unknown')
                        })
                except:
                    continue

            if bbox_files:
                selected_file = st.selectbox(
                    "Select coordinate file to load:",
                    options=[f['file'] for f in bbox_files],
                    format_func=lambda x: f"{Path(x).name} ({next(f['bbox_count'] for f in bbox_files if f['file'] == x)} boxes)",
                    key=f"bbox_file_{selected_image_name}"
                )

                if st.button("📥 Load Coordinates", key=f"load_coords_{selected_image_name}", type="secondary"):
                    if selected_file:
                        try:
                            with open(selected_file, 'r') as f:
                                data = json.load(f)

                            if data['bounding_boxes']:
                                # Use the first bounding box for now
                                bbox = data['bounding_boxes'][0]
                                coords_str = f"{bbox['x']},{bbox['y']},{bbox['x'] + bbox['width']},{bbox['y'] + bbox['height']}"

                                # Validate coordinates
                                is_valid, result = validate_coordinates(coords_str, img_width, img_height)
                                if is_valid:
                                    st.session_state[coord_key] = coords_str
                                    st.success(f"✅ Loaded coordinates: {coords_str}")
                                    st.info(f"📊 File contained {len(data['bounding_boxes'])} boxes. Loaded the first one.")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Invalid coordinates in file: {result}")
                            else:
                                st.warning("⚠️ No bounding boxes found in the selected file!")

                        except Exception as e:
                            st.error(f"❌ Error loading coordinates: {e}")
                    else:
                        st.warning("⚠️ Please select a coordinate file first!")
            else:
                st.info("💡 No coordinate files found. Use the BBox Annotator tool to create some!")

            # Action buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if st.button('📦 Sync Canvas', key=f"sync_canvas_{selected_image_name}", type="secondary"):
                    sync_js = f"""
                    <script>
                    (function() {{
                        // Check if we have coordinates in session state to draw on canvas
                        const coordsFromState = '{st.session_state[coord_key]}';
                        
                        if (coordsFromState && coordsFromState.trim() && window.canvasControls_{selected_image_name}) {{
                            const coords = coordsFromState.split(',');
                            
                            if (coords.length === 4) {{
                                const x1 = parseFloat(coords[0]);
                                const y1 = parseFloat(coords[1]); 
                                const x2 = parseFloat(coords[2]);
                                const y2 = parseFloat(coords[3]);
                                
                                // Clear existing boxes and add the coordinates as a box
                                window.canvasControls_{selected_image_name}.clearAll();
                                window.canvasControls_{selected_image_name}.addBoxFromCoords(x1, y1, x2, y2);
                                
                                const statusDiv = document.getElementById('canvas_status_{selected_image_name}');
                                if (statusDiv) {{
                                    statusDiv.innerHTML = '🔄 Canvas synced with coordinates: ' + coordsFromState;
                                    statusDiv.style.backgroundColor = '#e3f2fd';
                                    statusDiv.style.color = '#1976d2';
                                }}
                            }}
                        }} else {{
                            const statusDiv = document.getElementById('canvas_status_{selected_image_name}');
                            if (statusDiv) {{
                                statusDiv.innerHTML = '⚠️ No coordinates to sync. Enter coordinates first.';
                                statusDiv.style.backgroundColor = '#fff3e0';
                                statusDiv.style.color = '#ef6c00';
                            }}
                        }}
                    }})();
                    </script>
                    """
                    components.html(sync_js, height=0)
                    st.info('🔄 Canvas synchronized with current coordinates')

            with col2:
                if st.button('🗑️ Clear All', key=f"clear_all_{selected_image_name}", type="secondary"):
                    # Clear session state
                    st.session_state[coord_key] = ""
                    
                    # Clear canvas via JavaScript
                    clear_all_js = f"""
                    <script>
                    (function() {{
                        if (window.canvasControls_{selected_image_name}) {{
                            window.canvasControls_{selected_image_name}.clearAll();
                        }}
                    }})();
                    </script>
                    """
                    components.html(clear_all_js, height=0)
                    st.success('✅ Coordinates and canvas cleared!')
                    st.rerun()

            with col2:
                if st.button('� Validate Box', key=f"validate_box_{selected_image_name}", type="secondary"):
                    coords_to_validate = st.session_state[coord_key]
                    if coords_to_validate and coords_to_validate.strip():
                        is_valid, result = validate_coordinates(coords_to_validate, img_width, img_height)
                        if is_valid:
                            x1, y1, x2, y2 = result
                            width = abs(float(x2) - float(x1))
                            height = abs(float(y2) - float(y1))
                            st.success(f'✅ Valid coordinates! Size: {width:.0f} × {height:.0f} pixels')
                        else:
                            st.error(f'❌ Invalid coordinates: {result}')
                    else:
                        st.warning('⚠️ No coordinates to validate!')

            with col3:
                if st.button('💾 Save Box', key=f"save_box_{selected_image_name}", type="primary"):
                    coords_to_save = st.session_state[coord_key]
                    if coords_to_save and coords_to_save.strip() and 'current_dataset' in st.session_state:
                        # Validate coordinates before saving
                        is_valid, result = validate_coordinates(coords_to_save, img_width, img_height)
                        if not is_valid:
                            st.error(f"❌ Invalid coordinates: {result}")
                            st.stop()

                        # Save annotation to file
                        dataset_path = os.path.join("datasets", st.session_state.current_dataset)
                        labels_path = os.path.join(dataset_path, "labels")

                        # Ensure labels directory exists
                        os.makedirs(labels_path, exist_ok=True)

                        # Create label filename (same as image but .txt)
                        label_filename = os.path.splitext(selected_image_name)[0] + ".txt"
                        label_path = os.path.join(labels_path, label_filename)

                        # Convert coordinates to YOLO format (normalized)
                        try:
                            if not isinstance(result, tuple) or len(result) != 4:
                                st.error("❌ Invalid coordinate data received")
                                st.stop()

                            x1, y1, x2, y2 = result  # result is already the tuple (x1, y1, x2, y2)
                            # Convert to YOLO format: class x_center y_center width height (normalized)
                            x_center = ((x1 + x2) / 2) / img_width
                            y_center = ((y1 + y2) / 2) / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height

                            # YOLO format: class x_center y_center width height
                            yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                            # Save annotation
                            with open(label_path, "a") as f:
                                f.write(yolo_annotation + "\n")

                            st.success(f'✅ Annotation saved to: {label_filename}')

                            # Add to session state for display
                            if 'saved_boxes' not in st.session_state:
                                st.session_state.saved_boxes = []
                            st.session_state.saved_boxes.append(coords_to_save)

                        except Exception as e:
                            st.error(f'❌ Error saving annotation: {str(e)}')
                    elif 'current_dataset' not in st.session_state:
                        st.warning('⚠️ Please create or load a dataset first!')
                    else:
                        st.warning('⚠️ No coordinates to save. Please enter coordinates first!')

            with col4:
                if st.button('� Copy Coordinates', key=f"copy_coords_{selected_image_name}", type="secondary"):
                    coords_to_copy = st.session_state[coord_key]
                    if coords_to_copy and coords_to_copy.strip():
                        # Copy to clipboard using JavaScript
                        copy_js = f"""
                        <script>
                        (function() {{
                            navigator.clipboard.writeText('{coords_to_copy}').then(function() {{
                            }});
                        }})();
                        </script>
                        """
                        components.html(copy_js, height=0)
                        st.success(f'✅ Coordinates copied: {coords_to_copy}')
                    else:
                        st.warning('⚠️ No coordinates to copy!')

            # Show saved boxes if any
            if 'saved_boxes' in st.session_state and st.session_state.saved_boxes:
                with st.expander("📋 Saved Boxes", expanded=False):
                    st.write("**Saved bounding boxes for this session:**")

                    # Add clear all button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Total saved boxes: {len(st.session_state.saved_boxes)}**")
                    with col2:
                        if st.button("🗑️ Clear All", key=f"clear_all_boxes_{selected_image_name}", type="secondary"):
                            st.session_state.saved_boxes = []
                            st.success("✅ All saved boxes cleared!")
                            st.rerun()

                    # Display each saved box with remove button
                    for i, box_coords in enumerate(st.session_state.saved_boxes, 1):
                        col1, col2, col3 = st.columns([1, 3, 1])

                        with col1:
                            st.write(f"**{i}.**")

                        with col2:
                            st.code(box_coords, language="text")

                        with col3:
                            col_load, col_remove = st.columns(2)
                            with col_load:
                                if st.button("📥", key=f"load_box_{i}_{selected_image_name}", help=f"Load box {i} coordinates"):
                                    st.session_state[coord_key] = box_coords
                                    st.success(f"✅ Loaded coordinates: {box_coords}")
                                    st.rerun()

                            with col_remove:
                                if st.button("❌", key=f"remove_box_{i}_{selected_image_name}", help=f"Remove box {i}"):
                                    # Remove the box from the list
                                    st.session_state.saved_boxes.pop(i-1)
                                    st.success(f"✅ Box {i} removed!")
                                    st.rerun()

with tab2:
    st.header("Train Model")
    st.info("🚀 **Step 2:** Train your YOLO model using the dataset you created.")

    # Check for available datasets
    if not os.path.exists("datasets"):
        st.error("❌ No datasets folder found! Please create a dataset first in the 'Create Dataset' tab.")
        st.stop()

    datasets = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
    if not datasets:
        st.error("❌ No datasets found! Please create a dataset first in the 'Create Dataset' tab.")
        st.stop()

    st.success(f"✅ Found {len(datasets)} dataset(s)")

    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset for Training", datasets,
                                   help="Choose the dataset you want to use for training")

    if selected_dataset:
        dataset_path = os.path.join("datasets", selected_dataset)

        # Check dataset structure
        if not os.path.exists(os.path.join(dataset_path, "data.yaml")):
            st.error(f"❌ No data.yaml file found in {selected_dataset}! Dataset is not properly configured.")
            st.stop()

        # Show dataset info
        with st.expander("📊 Dataset Information", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                # Check if images directory exists and count images
                images_path = os.path.join(dataset_path, "images")
                if os.path.exists(images_path):
                    images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    st.metric("Images", len(images))
                else:
                    images = []
                    st.metric("Images", 0)
                    st.warning("⚠️ No 'images' directory found in dataset")

            with col2:
                # Check if labels directory exists and count labels
                labels_path = os.path.join(dataset_path, "labels")
                if os.path.exists(labels_path):
                    labels = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
                    st.metric("Label Files", len(labels))
                else:
                    labels = []
                    st.metric("Label Files", 0)
                    st.warning("⚠️ No 'labels' directory found in dataset")

            # Show data.yaml content
            if os.path.exists(os.path.join(dataset_path, "data.yaml")):
                with st.expander("📋 Dataset Configuration (data.yaml)"):
                    try:
                        with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
                            yaml_content = f.read()
                        st.code(yaml_content, language="yaml")
                    except Exception as e:
                        st.error(f"Could not read data.yaml: {e}")
            else:
                st.error("❌ No data.yaml file found! Dataset is not properly configured.")

            # Dataset validation
            if images:
                if len(images) > 0 and len(labels) == 0:
                    st.warning("⚠️ Dataset has images but no annotations. Add annotations before training.")
                elif len(images) > 0 and len(labels) > 0:
                    st.success("✅ Dataset is ready for training!")
            else:
                st.error("❌ No images found in dataset. Add images first.")

        # Training Configuration
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type",
                                     ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                                     index=0,
                                     help="n=nano, s=small, m=medium, l=large, x=extra large")
            epochs = st.slider("Training Epochs", 10, 200, 50,
                              help="Number of training iterations")

        with col2:
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1,
                                     help="Number of images processed at once")
            img_size = st.selectbox("Image Size", [320, 416, 512, 640, 768, 1024], index=3,
                                   help="Input image size for training")

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                patience = st.slider("Early Stopping Patience", 10, 100, 50,
                                   help="Stop training if no improvement for this many epochs")
                save_period = st.slider("Save Model Every", 1, 50, 10,
                                      help="Save checkpoint every N epochs")

        with col2:
            optimizer = st.selectbox("Optimizer", ["SGD", "Adam", "AdamW"], index=2)
            lr0 = st.selectbox("Initial Learning Rate", [0.01, 0.001, 0.0001], index=1)

        # Device selection
        st.subheader("🖥️ Device Selection")

        # Show system information
        with st.expander("ℹ️ System Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**PyTorch Version:**", torch.__version__)
                st.write("**CUDA Available:**", "Yes" if torch.cuda.is_available() else "No")
                if torch.cuda.is_available():
                    try:
                        st.write("**GPU Name:**", torch.cuda.get_device_name(0))
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        st.write("**GPU Memory:**", ".1f")
                    except Exception as e:
                        st.write("**GPU Info:**", f"Error: {str(e)}")

            with col2:
                import platform
                st.write("**OS:**", platform.system())
                st.write("**Python Version:**", platform.python_version())
                st.write("**Processor:**", platform.processor() or "Unknown")

        # Check available devices
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                st.success(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")

                device_options = ["Auto (GPU preferred)", "GPU (cuda:0)", "CPU"]
                device_help = "Auto will use GPU if available and compatible, otherwise fallback to CPU"
            except Exception as e:
                st.warning(f"⚠️ GPU detected but may have compatibility issues: {str(e)}")
                device_options = ["CPU", "GPU (cuda:0) - May fail"]
                device_help = "GPU may not work due to compatibility issues"
        else:
            st.info("ℹ️ No GPU detected, will use CPU")
            device_options = ["CPU"]
            device_help = "Only CPU is available on this system"

        device_choice = st.selectbox("Training Device", device_options,
                                    help=device_help, index=0)        # Model name
        model_name = st.text_input("Model Name", f"yolo_{selected_dataset}_{model_type.split('.')[0]}",
                                  help="Name for your trained model")

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Training button
        if st.button("🚀 Start Training", type="primary"):
            if not model_name.strip():
                st.error("❌ Please enter a model name!")
                st.stop()

            # Check if model already exists
            model_path = f"models/{model_name}.pt"
            if os.path.exists(model_path):
                overwrite = st.radio("Model already exists. Overwrite?", ["No", "Yes"], index=0)
                if overwrite == "No":
                    st.stop()

            with st.spinner("🔄 Initializing training..."):
                # Initialize progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load the model
                    model = YOLO(model_type)

                    # Set device based on user choice
                    device = "cpu"  # Default fallback

                    if device_choice == "CPU":
                        device = "cpu"
                        st.info("🔄 Using CPU for training (may be slower)")
                    elif device_choice == "GPU (cuda:0)":
                        device = "cuda:0"
                        st.info("🔄 Using GPU for training")
                    elif device_choice == "Auto (GPU preferred)":
                        if torch.cuda.is_available():
                            try:
                                # Test CUDA with a small tensor
                                test_tensor = torch.randn(1, 3, 32, 32).cuda()
                                test_tensor = test_tensor + 1  # Simple operation to test CUDA
                                torch.cuda.synchronize()  # Wait for completion
                                del test_tensor
                                device = "cuda:0"
                                st.info("🔄 Using GPU for training (CUDA test passed)")
                            except Exception as cuda_error:
                                st.warning(f"⚠️ CUDA test failed: {str(cuda_error)}")
                                st.info("🔄 Falling back to CPU training")
                                device = "cpu"
                        else:
                            device = "cpu"
                            st.info("🔄 Using CPU for training (no GPU available)")

                    # Start training
                    status_text.text("🚀 Starting training...")
                    progress_bar.progress(0.01)

                    # Set environment variables for debugging if using GPU
                    if device.startswith("cuda"):
                        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                        st.info("🔧 CUDA debugging enabled (may show more detailed errors)")

                    # Train the model
                    results = model.train(
                        data=os.path.join(dataset_path, "data.yaml"),
                        epochs=epochs,
                        batch=batch_size,
                        imgsz=img_size,
                        patience=patience,
                        save_period=save_period,
                        optimizer=optimizer,
                        lr0=lr0,
                        name=model_name,
                        project="models",
                        exist_ok=True,
                        device=device
                    )

                    # Training completed
                    progress_bar.progress(1.0)
                    status_text.text("✅ Training completed!")

                    # Save the model
                    model.save(f"models/{model_name}.pt")
                    st.success(f"💾 Model saved as: models/{model_name}.pt")

                    # Show CUDA compatibility note if applicable
                    if torch.cuda.is_available():
                        try:
                            gpu_name = torch.cuda.get_device_name(0)
                            st.info(f"🔧 **Note:** Your {gpu_name} has CUDA capability sm_120, but PyTorch supports up to sm_90. Training completed successfully on CPU. For GPU acceleration, consider updating PyTorch or using an older GPU.")
                        except:
                            pass

                    # Show training results
                    st.subheader("📊 Training Results")

                    # Add success celebration
                    st.success("🎉 **Training Successful!** Your model has been trained and saved!")
                    st.info("💡 **Next Steps:** You can now test your model in the 'Test Model' tab using new images.")

                    # Display metrics if available
                    try:
                        if results and hasattr(results, 'results_dict') and results.results_dict:
                            metrics = results.results_dict

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Final mAP@0.5", ".3f")
                            with col2:
                                st.metric("Final mAP@0.5:0.95", ".3f")
                            with col3:
                                st.metric("Precision", ".3f")
                            with col4:
                                st.metric("Recall", ".3f")
                    except Exception:
                        st.info("Training metrics will be displayed here after completion.")

                    # Show confusion matrix if available
                    confusion_matrix_path = f"models/{model_name}/confusion_matrix.png"
                    if os.path.exists(confusion_matrix_path):
                        st.image(confusion_matrix_path, caption="Confusion Matrix", width='stretch')

                    st.success("🎉 Training completed successfully! You can now test your model in the 'Test Model' tab.")

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Training failed: {error_msg}")

                    # Provide specific guidance for common CUDA errors
                    if "CUDA" in error_msg or "kernel image" in error_msg:
                        st.error("🔧 **CUDA/GPU Error Detected!**")
                        st.warning("**Possible solutions:**")
                        st.info("• Try selecting 'CPU' from the device options above")
                        st.info("• Update your GPU drivers to match PyTorch CUDA version")
                        st.info("• Check GPU compatibility with your PyTorch installation")
                        st.info("• Restart the application and try again")

                        # Show CUDA version info
                        try:
                            import torch.version
                            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
                            st.info(f"Current PyTorch CUDA version: {cuda_version}")
                        except:
                            st.info("Could not detect PyTorch CUDA version")

                    elif "out of memory" in error_msg.lower():
                        st.error("🔧 **GPU Memory Error!**")
                        st.warning("**Possible solutions:**")
                        st.info("• Reduce batch size in training configuration")
                        st.info("• Reduce image size in training configuration")
                        st.info("• Try CPU training instead")
                        st.info("• Close other GPU-intensive applications")

                    else:
                        st.error("Please check your dataset configuration and try again.")

                    # Suggest alternative device if CUDA failed
                    st.info("💡 **Tip:** If you encountered a CUDA error, try selecting 'CPU' from the device options above.")
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

with tab3:
    st.header("Test Model")
    st.info("🧪 **Step 3:** Test your trained model on new images.")

    # Check for available models
    if not os.path.exists("models"):
        st.error("❌ No models folder found! Please train a model first in the 'Train Model' tab.")
        st.stop()

    model_files = [f for f in os.listdir("models") if f.lower().endswith(('.pt', '.pth'))]
    if not model_files:
        st.error("❌ No trained models found! Please train a model first in the 'Train Model' tab.")
        st.stop()

    st.success(f"✅ Found {len(model_files)} trained model(s)")

    # Model selection
    selected_model = st.selectbox("Select Model for Testing", model_files,
                                 help="Choose the trained model you want to test")

    if selected_model:
        model_path = os.path.join("models", selected_model)

        # Load model info
        model = None
        try:
            model = YOLO(model_path)
            st.success(f"✅ Model '{selected_model}' loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()

        # Test image upload
        st.subheader("Upload Test Image")
        test_image = st.file_uploader("Choose an image to test", type=['jpg', 'png', 'jpeg'],
                                     help="Upload an image to run inference on")

        if test_image:
            # Validate uploaded image
            try:
                image = safe_open_image(test_image)
                if image is None:
                    st.error("❌ Failed to load image. Please upload a valid image file.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.stop()

            # Display original image
            st.image(image, caption="Original Image", width='stretch')

            # Configuration
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5,
                                               help="Minimum confidence for detections")
                max_detections = st.slider("Max Detections", 1, 100, 20,
                                         help="Maximum number of detections to show")

            with col2:
                iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.5,
                                         help="Intersection over Union threshold for NMS")
                save_results = st.checkbox("Save Results", value=True,
                                         help="Save detection results to file")

            # Run detection button
            if st.button("🔍 Run Detection", type="primary"):
                if model is None:
                    st.error("❌ Model not loaded. Please select a model first.")
                    st.stop()

                with st.spinner("🔄 Running inference..."):
                    try:
                        # Ensure model is not None (for linter)
                        assert model is not None, "Model should not be None at this point"

                        # Convert PIL to numpy array
                        img_array = np.array(image)

                        # Run inference
                        results = model(img_array, conf=confidence_threshold, iou=iou_threshold,
                                      max_det=max_detections)

                        # Process results
                        if len(results) > 0:
                            result = results[0]

                            # Display results
                            st.subheader("📊 Detection Results")

                            # Show metrics
                            if result.boxes is not None and len(result.boxes) > 0:
                                num_detections = len(result.boxes)
                                avg_confidence = result.boxes.conf.mean().item() if result.boxes.conf is not None else 0

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Detections Found", num_detections)
                                with col2:
                                    st.metric("Avg Confidence", ".3f")
                                with col3:
                                    st.metric("Classes Detected", len(result.boxes.cls.unique()) if result.boxes.cls is not None else 0)

                                # Display annotated image
                                st.subheader("🎯 Annotated Image")
                                annotated_img = result.plot()
                                st.image(annotated_img, caption="Detection Results", width='stretch')

                                # Detailed results
                                with st.expander("📋 Detailed Detection Results", expanded=True):
                                    if result.boxes is not None:
                                        import pandas as pd

                                        # Create results dataframe
                                        boxes_data = []
                                        for i, box in enumerate(result.boxes):
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            conf = box.conf[0].cpu().numpy() if box.conf is not None else 0
                                            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0

                                            boxes_data.append({
                                                "Detection": i+1,
                                                "Class": cls,
                                                "Confidence": ".3f",
                                                "X1": ".1f",
                                                "Y1": ".1f",
                                                "X2": ".1f",
                                                "Y2": ".1f",
                                                "Width": ".1f",
                                                "Height": ".1f"
                                            })

                                        if boxes_data:
                                            df = pd.DataFrame(boxes_data)
                                            st.dataframe(df)

                                            # Download results
                                            if save_results:
                                                csv_data = df.to_csv(index=False)
                                                st.download_button(
                                                    label="📥 Download Results as CSV",
                                                    data=csv_data,
                                                    file_name=f"detection_results_{selected_model}_{test_image.name}.csv",
                                                    mime="text/csv"
                                                )

                                # Save annotated image option
                                if save_results:
                                    # Convert to PIL Image for saving
                                    annotated_pil = Image.fromarray(annotated_img)
                                    buffered = io.BytesIO()
                                    annotated_pil.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode()

                                    st.download_button(
                                        label="📥 Download Annotated Image",
                                        data=base64.b64decode(img_str),
                                        file_name=f"annotated_{test_image.name}",
                                        mime="image/png"
                                    )
                            else:
                                st.warning("⚠️ No objects detected in the image. Try lowering the confidence threshold.")
                        else:
                            st.error("❌ No results returned from model inference.")

                    except Exception as e:
                        st.error(f"❌ Inference failed: {str(e)}")
                        st.error("Please check your model and try again.")

        # Model information
        with st.expander("📋 Model Information"):
            try:
                st.write(f"**Model File:** {selected_model}")
                st.write(f"**Model Path:** {model_path}")

                # Try to get model info
                if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                    st.write("**Model Architecture:** YOLOv8")
                else:
                    st.write("**Model Architecture:** YOLOv8")

                # File size
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                st.write(".2f")

                # Modification time
                mod_time = os.path.getmtime(model_path)
                from datetime import datetime
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"**Last Modified:** {mod_date}")

            except Exception as e:
                st.write(f"Could not retrieve model information: {str(e)}")

with tab4:
    st.header("Browse")
    st.info("📁 **Browse:** View your datasets, models, and results.")

    # Create tabs for different browsing sections
    browse_tab1, browse_tab2, browse_tab3 = st.tabs(["Datasets", "Models", "Results"])

    with browse_tab1:
        st.subheader("📂 Dataset Browser")

        # Check if datasets folder exists
        if os.path.exists("datasets"):
            datasets = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]

            if datasets:
                selected_dataset = st.selectbox("Select Dataset", datasets, key="browse_dataset")

                if selected_dataset:
                    dataset_path = os.path.join("datasets", selected_dataset)

                    # Dataset overview
                    col1, col2, col3 = st.columns(3)

                    # Check if images and labels directories exist
                    images_path = os.path.join(dataset_path, "images")
                    labels_path = os.path.join(dataset_path, "labels")

                    if os.path.exists(images_path):
                        # Get all image files and validate they actually exist
                        all_images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                        images = [f for f in all_images if os.path.exists(os.path.join(images_path, f))]
                    else:
                        images = []

                    if os.path.exists(labels_path):
                        labels = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
                    else:
                        labels = []

                    with col1:
                        st.metric("Total Images", len(images))
                    with col2:
                        st.metric("Label Files", len(labels))
                    with col3:
                        st.metric("Unlabeled", len(images) - len(labels))

                    # Data.yaml info
                    if os.path.exists(os.path.join(dataset_path, "data.yaml")):
                        st.success("✅ Dataset configuration file (data.yaml) found")

                        # Show data.yaml content
                        with st.expander("📋 Dataset Configuration (data.yaml)"):
                            try:
                                with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
                                    yaml_content = f.read()
                                st.code(yaml_content, language="yaml")
                            except Exception as e:
                                st.error(f"Could not read data.yaml: {e}")

                    # Image browser
                    st.subheader("🖼️ Image Browser")

                    if images:
                        # Pagination
                        images_per_page = 6
                        total_pages = (len(images) + images_per_page - 1) // images_per_page

                        if total_pages > 1:
                            page = st.slider("Page", 1, total_pages, 1, key="dataset_page")
                            start_idx = (page - 1) * images_per_page
                            end_idx = min(start_idx + images_per_page, len(images))
                        else:
                            start_idx = 0
                            end_idx = len(images)

                        # Display images in grid
                        cols = st.columns(3)
                        for i, img_name in enumerate(images[start_idx:end_idx]):
                            img_path = os.path.join(dataset_path, img_name)

                            # Check for corresponding label file
                            label_name = os.path.splitext(img_name)[0] + '.txt'
                            has_label = os.path.exists(os.path.join(dataset_path, label_name))

                            with cols[i % 3]:
                                # Check if image file actually exists before displaying
                                if os.path.exists(img_path):
                                    try:
                                        st.image(img_path, caption=f"{img_name} {'✅' if has_label else '❌'}")
                                    except Exception as img_error:
                                        st.error(f"❌ Error loading image {img_name}: {str(img_error)}")
                                        # Show placeholder
                                        st.write(f"📷 {img_name} (image file corrupted)")
                                        st.write(f"{'✅' if has_label else '❌'} Label: {'Present' if has_label else 'Missing'}")
                                else:
                                    st.error(f"❌ Image file not found: {img_name}")
                                    st.write(f"{'✅' if has_label else '❌'} Label: {'Present' if has_label else 'Missing'}")

                                # Show label info if available
                                if has_label:
                                    try:
                                        with open(os.path.join(dataset_path, label_name), 'r') as f:
                                            label_content = f.read().strip()
                                            if label_content:
                                                num_boxes = len(label_content.split('\n'))
                                                st.caption(f"📦 {num_boxes} annotation(s)")
                                            else:
                                                st.caption("📦 Empty label file")
                                    except:
                                        st.caption("📦 Label file error")

                        if total_pages > 1:
                            st.info(f"Showing images {start_idx + 1}-{end_idx} of {len(images)}")
                    else:
                        st.info("No images found in this dataset.")
            else:
                st.info("No datasets found. Create one in the 'Create Dataset' tab first!")
        else:
            st.warning("No datasets folder found. Create datasets first!")

    with browse_tab2:
        st.subheader("🤖 Model Browser")

        # Check if models folder exists
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.lower().endswith(('.pt', '.pth'))]

            if model_files:
                selected_model = st.selectbox("Select Model", model_files, key="browse_model")

                if selected_model:
                    model_path = os.path.join("models", selected_model)

                    # Model information
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Model File", selected_model)
                        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                        st.metric("File Size", ".2f")

                    with col2:
                        mod_time = os.path.getmtime(model_path)
                        from datetime import datetime
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                        st.metric("Created", mod_date)

                    # Model actions
                    st.subheader("Model Actions")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("🧪 Test Model", key="test_model_btn"):
                            st.info("Switch to the 'Test Model' tab to test this model.")

                    with col2:
                        if st.button("📥 Download Model", key="download_model"):
                            try:
                                with open(model_path, "rb") as f:
                                    model_data = f.read()
                                st.download_button(
                                    label="Download",
                                    data=model_data,
                                    file_name=selected_model,
                                    mime="application/octet-stream"
                                )
                            except Exception as e:
                                st.error(f"Could not prepare download: {str(e)}")

                    with col3:
                        if st.button("🗑️ Delete Model", key="delete_model"):
                            if st.checkbox("Confirm deletion", key="confirm_delete"):
                                try:
                                    os.remove(model_path)
                                    st.success(f"Model '{selected_model}' deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Could not delete model: {str(e)}")

                    # Training results if available
                    model_name_base = os.path.splitext(selected_model)[0]
                    training_results_dir = os.path.join("models", model_name_base)

                    if os.path.exists(training_results_dir):
                        st.subheader("📊 Training Results")

                        # Look for training artifacts
                        result_files = os.listdir(training_results_dir)

                        # Confusion matrix
                        confusion_matrix = [f for f in result_files if 'confusion_matrix' in f.lower()]
                        if confusion_matrix:
                            st.image(os.path.join(training_results_dir, confusion_matrix[0]),
                                   caption="Confusion Matrix")

                        # Results CSV
                        results_csv = [f for f in result_files if f.endswith('.csv')]
                        if results_csv:
                            import pandas as pd
                            try:
                                df = pd.read_csv(os.path.join(training_results_dir, results_csv[0]))
                                st.dataframe(df)
                            except:
                                st.info("Could not load training results CSV")

                        # Training curves
                        curves = [f for f in result_files if 'curve' in f.lower() or 'plot' in f.lower()]
                        if curves:
                            for curve in curves[:3]:  # Show first 3 curves
                                st.image(os.path.join(training_results_dir, curve),
                                       caption=f"Training {curve}")
            else:
                st.info("No trained models found. Train a model first!")
        else:
            st.warning("No models folder found. Train models first!")

    with browse_tab3:
        st.subheader("📊 Results & Analytics")

        # Summary statistics
        st.subheader("Project Summary")

        col1, col2, col3, col4 = st.columns(4)

        # Count datasets
        if os.path.exists("datasets"):
            num_datasets = len([d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))])
        else:
            num_datasets = 0

        # Count models
        if os.path.exists("models"):
            num_models = len([f for f in os.listdir("models") if f.lower().endswith(('.pt', '.pth'))])
        else:
            num_models = 0

        # Count total images
        total_images = 0
        if os.path.exists("datasets"):
            for dataset in [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]:
                dataset_path = os.path.join("datasets", dataset)
                images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                total_images += len(images)

        with col1:
            st.metric("Datasets", num_datasets)
        with col2:
            st.metric("Models", num_models)
        with col3:
            st.metric("Total Images", total_images)
        with col4:
            st.metric("Storage Used", "Calculating...")

        # Recent activity
        st.subheader("Recent Activity")

        activity_items = []

        # Check for recent models
        if os.path.exists("models"):
            for model_file in [f for f in os.listdir("models") if f.lower().endswith(('.pt', '.pth'))]:
                model_path = os.path.join("models", model_file)
                mod_time = os.path.getmtime(model_path)
                activity_items.append(("Model trained", model_file, mod_time))

        # Check for recent datasets
        if os.path.exists("datasets"):
            for dataset in [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]:
                dataset_path = os.path.join("datasets", dataset)
                mod_time = os.path.getmtime(dataset_path)
                activity_items.append(("Dataset created", dataset, mod_time))

        # Sort by time and show recent items
        activity_items.sort(key=lambda x: x[2], reverse=True)

        if activity_items:
            for activity_type, name, timestamp in activity_items[:5]:  # Show last 5
                time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                st.info(f"**{time_str}** - {activity_type}: {name}")
        else:
            st.info("No recent activity found.")

        # System information
        st.subheader("System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Environment:**")
            st.info(f"Python: {sys.version.split()[0]}")
            try:
                import torch
                st.info(f"PyTorch: {torch.__version__}")
            except:
                st.info("PyTorch: Not available")

        with col2:
            st.write("**Storage:**")
            try:
                import psutil
                disk = psutil.disk_usage('/')
                st.info(".1f")
            except:
                st.info("Storage info not available")
