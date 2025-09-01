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

st.info("🚀 **Welcome!** This web UI helps you create YOLO v8 datasets, train models, and test them on images. Use the tabs below to navigate through the workflow.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Create Dataset", "Train Model", "Test Model", "Browse"])

with tab1:
    st.header("Create Dataset")
    st.info("📸 **Step 1:** Upload images and create your YOLO dataset. For annotation, use the dedicated annotation interface in the 'Browse' tab.")

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
        - Use the annotation tools in the 'Browse' tab for best results

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

    # File uploader
    uploaded_files = st.file_uploader('Upload images', accept_multiple_files=True, type=['jpg','png','jpeg'])

    if uploaded_files:
        # Display upload summary
        st.success(f'✅ Uploaded {len(uploaded_files)} files!')

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

        # Image selector for preview
        if uploaded_files:
            image_names = [file.name for file in uploaded_files]
            selected_image_name = st.selectbox('Preview uploaded image:', image_names)

            if selected_image_name:
                selected_file = next(file for file in uploaded_files if file.name == selected_image_name)
                image = safe_open_image(selected_file)
                if image:
                    st.image(image, caption=f"Preview: {selected_image_name}", use_column_width=True)
                    st.info(f"📏 **Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                else:
                    st.error("❌ Failed to load image preview.")

    # Annotation instructions
    st.markdown("---")
    st.subheader("🎯 Next Step: Annotation")
    st.info("""
    **📋 Your images are uploaded! Now you need to annotate them:**
    
    1. **Go to the "Browse" tab** (View Dataset)
    2. **Select your dataset** from the dropdown
    3. **Choose an image** to annotate
    4. **Draw bounding boxes** on your objects
    5. **Annotations are automatically saved** to YOLO format
    
    ✨ The streamlined annotation system saves directly to your dataset files!
    """)
    
    # Quick links
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔗 Quick Actions:**")
        st.markdown("- Switch to **Browse** tab to start annotating")
        st.markdown("- Your dataset is ready for annotation")
        st.markdown("- All annotations save automatically")
    
    with col2:
        if 'current_dataset' in st.session_state:
            st.success(f"📂 **Current Dataset:** {st.session_state.current_dataset}")
            st.markdown("✅ Ready for annotation!")
        else:
            st.info("💡 Create a dataset above to get started")

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
                        st.write("**GPU Memory:**", f"{gpu_memory:.1f} GB")
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
                                    help=device_help, index=0)

        # Model name
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
                try:
                    # Load the YOLO model
                    model = YOLO(model_type)
                    
                    # Set device
                    if device_choice.startswith("GPU"):
                        device = "cuda:0"
                    elif device_choice.startswith("Auto"):
                        device = "auto"
                    else:
                        device = "cpu"
                    
                    st.info(f"🎯 Training device: {device}")
                    
                    # Get absolute path to data.yaml
                    data_yaml_path = os.path.abspath(os.path.join(dataset_path, "data.yaml"))
                    
                    st.success("✅ Model loaded successfully!")
                    st.info("🚀 Starting training... This may take a while.")
                    
                    # Create a progress placeholder
                    progress_placeholder = st.empty()
                    
                    # Train the model
                    results = model.train(
                        data=data_yaml_path,
                        epochs=epochs,
                        batch=batch_size,
                        imgsz=img_size,
                        device=device,
                        patience=patience,
                        save_period=save_period,
                        optimizer=optimizer,
                        lr0=lr0,
                        project="models",
                        name=model_name,
                        exist_ok=True
                    )
                    
                    # Training completed
                    st.success("🎉 Training completed successfully!")
                    
                    # Show training results
                    st.subheader("📊 Training Results")
                    
                    # Check if results folder exists
                    results_path = os.path.join("models", model_name)
                    if os.path.exists(results_path):
                        # Show training charts if available
                        charts = ["results.png", "confusion_matrix.png", "BoxF1_curve.png", "BoxPR_curve.png"]
                        
                        for chart in charts:
                            chart_path = os.path.join(results_path, chart)
                            if os.path.exists(chart_path):
                                st.image(chart_path, caption=chart)
                        
                        # Show best model info
                        best_model_path = os.path.join(results_path, "weights", "best.pt")
                        if os.path.exists(best_model_path):
                            st.success(f"✅ Best model saved to: {best_model_path}")
                            
                            # Copy best model to models directory for easy access
                            import shutil
                            final_model_path = f"models/{model_name}.pt"
                            shutil.copy2(best_model_path, final_model_path)
                            st.info(f"📂 Model also saved as: {final_model_path}")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.error("Please check your dataset and try again.")

with tab3:
    st.header("Test Model")
    st.info("🧪 **Step 3:** Test your trained YOLO model on new images.")

    # Check for available models
    if not os.path.exists("models"):
        st.error("❌ No models folder found! Please train a model first.")
        st.stop()

    # Get list of model files
    model_files = [f for f in os.listdir("models") if f.endswith('.pt')]
    
    if not model_files:
        st.error("❌ No trained models found! Please train a model first.")
        st.stop()

    st.success(f"✅ Found {len(model_files)} trained model(s)")

    # Model selection
    selected_model = st.selectbox("Select Trained Model", model_files,
                                 help="Choose the model you want to use for testing")

    if selected_model:
        model_path = os.path.join("models", selected_model)
        
        # Load model info
        try:
            model = YOLO(model_path)
            st.success(f"✅ Loaded model: {selected_model}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()

        # Test image upload
        st.subheader("📷 Upload Test Images")
        test_files = st.file_uploader('Upload test images', accept_multiple_files=True, type=['jpg','png','jpeg'])

        if test_files:
            st.success(f'✅ Uploaded {len(test_files)} test image(s)!')

            # Test configuration
            st.subheader("⚙️ Detection Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                                     help="Minimum confidence for detections")
                iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05,
                                        help="Intersection over Union threshold for NMS")
            
            with col2:
                max_detections = st.slider("Max Detections", 1, 1000, 100,
                                         help="Maximum number of detections per image")
                img_size_test = st.selectbox("Image Size", [320, 416, 512, 640, 768, 1024], index=3,
                                           help="Image size for inference")

            # Process images
            if st.button("🔍 Run Detection", type="primary"):
                with st.spinner("🔄 Running detection on images..."):
                    for i, test_file in enumerate(test_files):
                        st.subheader(f"📸 Results for: {test_file.name}")
                        
                        # Load and display original image
                        image = safe_open_image(test_file)
                        if image is None:
                            st.error(f"❌ Failed to load {test_file.name}")
                            continue
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Image:**")
                            st.image(image, caption=f"Original: {test_file.name}", use_column_width=True)
                        
                        try:
                            # Run detection
                            results = model(image, 
                                          conf=confidence,
                                          iou=iou_threshold,
                                          max_det=max_detections,
                                          imgsz=img_size_test)
                            
                            with col2:
                                st.write("**Detection Results:**")
                                
                                # Get detection results
                                if len(results) > 0 and len(results[0].boxes) > 0:
                                    # Annotate image
                                    annotated_image = results[0].plot()
                                    
                                    # Convert BGR to RGB (OpenCV to PIL)
                                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                                    annotated_pil = Image.fromarray(annotated_image_rgb)
                                    
                                    st.image(annotated_pil, caption=f"Detections: {test_file.name}", use_column_width=True)
                                    
                                    # Show detection details
                                    st.write(f"**🎯 Found {len(results[0].boxes)} detection(s):**")
                                    
                                    for j, box in enumerate(results[0].boxes):
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])
                                        
                                        # Get class name
                                        class_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                                        
                                        st.write(f"• **Detection {j+1}:** {class_name} (Confidence: {conf:.2f})")
                                else:
                                    st.image(image, caption=f"No detections: {test_file.name}", use_column_width=True)
                                    st.info("🔍 No objects detected")
                            
                            # Download option for annotated image
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                annotated_image = results[0].plot()
                                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                                annotated_pil = Image.fromarray(annotated_image_rgb)
                                
                                # Convert to bytes for download
                                buf = io.BytesIO()
                                annotated_pil.save(buf, format='PNG')
                                buf.seek(0)
                                
                                st.download_button(
                                    label=f"⬇️ Download Annotated {test_file.name}",
                                    data=buf.getvalue(),
                                    file_name=f"detected_{test_file.name}",
                                    mime="image/png",
                                    key=f"download_{i}"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Detection failed for {test_file.name}: {str(e)}")
                        
                        # Add separator between images
                        if i < len(test_files) - 1:
                            st.markdown("---")

with tab4:
    st.header("Browse Datasets")
    st.info("🗂️ **Browse and annotate your datasets.** Select a dataset to view and annotate images.")

    # Check for datasets
    if not os.path.exists("datasets"):
        st.error("❌ No datasets folder found! Please create a dataset first.")
        st.stop()

    datasets = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
    if not datasets:
        st.error("❌ No datasets found! Please create a dataset first.")
        st.stop()

    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset", datasets, key="browse_dataset_selector")

    if selected_dataset:
        dataset_path = os.path.join("datasets", selected_dataset)
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        # Dataset stats
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)

        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))] if os.path.exists(images_path) else []
        labels = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')] if os.path.exists(labels_path) else []

        with col1:
            st.metric("Total Images", len(images))
        with col2:
            st.metric("Label Files", len(labels))
        with col3:
            # Count images with annotations
            annotated_count = 0
            for image_file in images:
                label_file = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.join(labels_path, label_file)
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        if f.read().strip():  # Has content
                            annotated_count += 1
            st.metric("Annotated Images", annotated_count)

        if images:
            # Image browser
            st.subheader("🖼️ Image Browser")
            
            # Image selection
            selected_image = st.selectbox("Select Image", images, key="browse_image_selector")
            
            if selected_image:
                image_path = os.path.join(images_path, selected_image)
                image = safe_open_image(image_path)
                
                if image:
                    st.subheader(f"📝 Annotating: {selected_image}")
                    
                    # Display current annotations count
                    label_file = os.path.splitext(selected_image)[0] + ".txt"
                    label_path = os.path.join(labels_path, label_file)
                    
                    annotation_count = 0
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                annotation_count = len([line for line in content.split('\n') if line.strip()])
                    
                    st.info(f"📦 Current annotations: {annotation_count}")
                    
                    # Annotation interface - simplified direct save system
                    img_width, img_height = image.size
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Enhanced annotation canvas with direct saving
                    annotation_html = f"""
                    <div style="position: relative; display: inline-block; border: 2px solid #ddd; margin: 10px 0;">
                        <img id="dataset_image_{selected_image}" src="data:image/png;base64,{img_base64}"
                             style="display: block; max-width: 100%; height: auto;" />
                        <canvas id="dataset_canvas_{selected_image}"
                                style="position: absolute; top: 0; left: 0; cursor: crosshair; border: none; pointer-events: auto; z-index: 10;"
                                width="{img_width}" height="{img_height}">
                        </canvas>
                    </div>

                    <div id="coord_status_{selected_image}" style="margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;">
                        👆 Click and drag on the image above to draw a bounding box
                    </div>
                    
                    <script>
                        (function() {{
                            const img = document.getElementById('dataset_image_{selected_image}');
                            const canvas = document.getElementById('dataset_canvas_{selected_image}');
                            const statusDiv = document.getElementById('coord_status_{selected_image}');
                            
                            if (!img || !canvas || !statusDiv) {{
                                console.error('Elements not found for {selected_image}');
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
                                canvas.width = {img_width};
                                canvas.height = {img_height};
                            }}

                            // Initialize canvas
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
                            }});

                            canvas.addEventListener('mousemove', function(e) {{
                                if (!isDrawing) return;
                                e.preventDefault();
                                
                                const coords = getCanvasCoords(e.clientX, e.clientY);
                                const width = coords.x - startX;
                                const height = coords.y - startY;
                                
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
                                    
                                    // Auto-save annotation
                                    saveAnnotation(box);
                                    
                                    updateStatus('✅ Box saved automatically!');
                                }} else {{
                                    updateStatus('Box too small! Try again.');
                                }}
                                
                                redrawBoxes();
                            }});

                            // Auto-save function
                            function saveAnnotation(box) {{
                                // Convert to YOLO format
                                const centerX = ((box.x * 2 + box.width) / 2) / {img_width};
                                const centerY = ((box.y * 2 + box.height) / 2) / {img_height};
                                const width = box.width / {img_width};
                                const height = box.height / {img_height};
                                
                                const yoloLine = `0 ${{centerX.toFixed(6)}} ${{centerY.toFixed(6)}} ${{width.toFixed(6)}} ${{height.toFixed(6)}}`;
                                
                                // Store for Python to pick up
                                localStorage.setItem('new_annotation_{selected_image}', yoloLine);
                                localStorage.setItem('annotation_timestamp_{selected_image}', Date.now());
                                
                                console.log('Annotation saved:', yoloLine);
                            }}

                            // Clear function
                            window.clearDatasetCanvas_{selected_image} = function() {{
                                ctx.clearRect(0, 0, canvas.width, canvas.height);
                                savedBoxes = [];
                                updateStatus('👆 Click and drag on the image above to draw a bounding box');
                            }};

                            updateStatus('Ready! Click and drag to draw boxes.');
                        }})();
                    </script>
                    """
                    
                    components.html(annotation_html, height=img_height + 120)
                    
                    # Auto-save handler
                    annotation_data = st.text_input(
                        "Annotation handler",
                        key=f"annotation_handler_{selected_image}",
                        placeholder="",
                        label_visibility="hidden"
                    )
                    
                    # Check for new annotations via JavaScript
                    check_annotations_script = f"""
                    <script>
                    (function() {{
                        function checkForNewAnnotations() {{
                            const newAnnotation = localStorage.getItem('new_annotation_{selected_image}');
                            const timestamp = localStorage.getItem('annotation_timestamp_{selected_image}');
                            
                            if (newAnnotation && timestamp) {{
                                // Send to Streamlit
                                const input = document.querySelector('input[aria-label="Annotation handler"]');
                                if (input) {{
                                    input.value = newAnnotation;
                                    input.dispatchEvent(new Event('input', {{bubbles: true}}));
                                    input.dispatchEvent(new Event('change', {{bubbles: true}}));
                                }}
                                
                                // Clear localStorage
                                localStorage.removeItem('new_annotation_{selected_image}');
                                localStorage.removeItem('annotation_timestamp_{selected_image}');
                            }}
                        }}
                        
                        // Check every 500ms for new annotations
                        setInterval(checkForNewAnnotations, 500);
                    }})();
                    </script>
                    """
                    
                    components.html(check_annotations_script, height=0)
                    
                    # Process new annotations
                    if annotation_data and annotation_data.strip():
                        try:
                            # Ensure labels directory exists
                            os.makedirs(labels_path, exist_ok=True)
                            
                            # Append annotation to label file
                            with open(label_path, "a") as f:
                                f.write(annotation_data + "\n")
                            
                            st.success(f"✅ Annotation saved! Total annotations: {annotation_count + 1}")
                            
                            # Clear the input
                            st.session_state[f"annotation_handler_{selected_image}"] = ""
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error saving annotation: {str(e)}")
                    
                    # Manual controls
                    st.subheader("🎛️ Controls")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button('🗑️ Clear Canvas', key=f"clear_dataset_canvas_{selected_image}", type="secondary"):
                            clear_js = f"""
                            <script>
                            if (typeof window.clearDatasetCanvas_{selected_image} === 'function') {{
                                window.clearDatasetCanvas_{selected_image}();
                            }}
                            </script>
                            """
                            components.html(clear_js, height=0)
                            st.success('✅ Canvas cleared!')
                    
                    with col2:
                        if st.button('📄 View Annotations', key=f"view_annotations_{selected_image}", type="secondary"):
                            if os.path.exists(label_path):
                                with open(label_path, 'r') as f:
                                    content = f.read().strip()
                                if content:
                                    st.code(content, language="text")
                                else:
                                    st.info("No annotations found.")
                            else:
                                st.info("No annotation file found.")
                    
                    with col3:
                        if st.button('🗑️ Clear All Annotations', key=f"clear_annotations_{selected_image}", type="secondary"):
                            if st.checkbox(f"Confirm clear all annotations for {selected_image}", key=f"confirm_clear_{selected_image}"):
                                if os.path.exists(label_path):
                                    with open(label_path, 'w') as f:
                                        f.write("")
                                    st.success("✅ All annotations cleared!")
                                    st.rerun()
                
                else:
                    st.error("❌ Failed to load image.")
        else:
            st.info("📷 No images found in this dataset.")
