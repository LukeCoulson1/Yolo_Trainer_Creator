#!/usr/bin/env python3
"""
YOLO Dataset Creator & Trainer
A comprehensive GUI application for creating YOLO datasets, annotating images,
training models, and testing them - all in one place.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import json
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import platform

class DatasetCreator:
        """Start model training"""
        if self.training_active:
            messagebox.showwarning("Warning", "Training is already in progress")
            return

        dataset_name = self.train_dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset for training")
            return

        model_name = self.available_models.get(self.model_var.get())
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model")
            return

        # Check if dataset exists and has images
        dataset_path = os.path.join("datasets", dataset_name)
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        if not os.path.exists(images_path):
            messagebox.showerror("Error", "Dataset has no images folder")
            return

        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            messagebox.showerror("Error", "Dataset has no images")
            return

        # Start training in background thread
        self.training_active = True
        self.training_thread = threading.Thread(target=self._train_model,
                                              args=(dataset_path, model_name),
                                              daemon=True)
        self.training_thread.start()

        self.log_training("🚀 Starting training...")
        self.training_status.set("Training in progress...")

    def stop_training(self):
        """Stop current training"""
        if self.training_active:
            self.training_active = False
            self.log_training("⏹️ Training stopped by user")
            self.training_status.set("Training stopped")

    def view_training_results(self):
        """View training results"""
        # This would open results folder or display metrics
        messagebox.showinfo("Info", "Training results viewer not implemented yet")

    def browse_model(self):
        """Browse for model file"""
        filetypes = [('PyTorch model files', '*.pt'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=filetypes
        )
        if filename:
            self.test_model_path.set(filename)

    def load_test_image(self):
        """Load test image for detection"""
        filetypes = [('Image files', '*.jpg *.jpeg *.png *.bmp'), ('All files', '*.*')]
        filename = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=filetypes
        )
        if filename:
            try:
                self.test_image = Image.open(filename)
                self.display_test_image()
                self.log_status("🖼️ Test image loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def run_detection(self):
        """Run detection on test image"""
        model_path = self.test_model_path.get()
        if not model_path:
            messagebox.showwarning("Warning", "Please select a model file")
            return

        if not hasattr(self, 'test_image'):
            messagebox.showwarning("Warning", "Please load a test image first")
            return

        try:
            # Load model
            self.log_status("🔄 Loading model...")
            model = YOLO(model_path)

            # Run detection
            self.log_status("🔍 Running detection...")
            results = model(self.test_image)

            # Display results
            self.display_detection_results(results[0])

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.log_status(f"❌ Detection failed: {str(e)}")

    def display_test_image(self):
        """Display test image on canvas"""
        if not hasattr(self, 'test_image'):
            return

        # Clear canvas
        self.test_canvas.delete("all")

        # Get canvas size
        canvas_width = self.test_canvas.winfo_width()
        canvas_height = self.test_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_test_image)
            return

        # Calculate scaling
        img_width, img_height = self.test_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y)

        # Calculate display size
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        # Calculate offset
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        # Resize image
        resized_image = self.test_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.test_photo = ImageTk.PhotoImage(resized_image)

        # Display image
        self.test_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.test_photo)

    def display_detection_results(self, results):
        """Display detection results"""
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Display results text
        self.results_text.insert(tk.END, f"Detection Results:\n")
        self.results_text.insert(tk.END, f"Found {len(results.boxes)} objects\n\n")

        if len(results.boxes) > 0:
            for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                x1, y1, x2, y2 = box.tolist()
                confidence = conf.item()
                class_id = int(cls.item())

                self.results_text.insert(tk.END, f"Object {i+1}:\n")
                self.results_text.insert(tk.END, f"  Class: {class_id}\n")
                self.results_text.insert(tk.END, f"  Confidence: {confidence:.3f}\n")
                self.results_text.insert(tk.END, f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n\n")

        self.results_text.config(state=tk.DISABLED)

        # Draw bounding boxes on canvas
        self.draw_detection_boxes(results)

        self.log_status(f"✅ Detection complete - found {len(results.boxes)} objects")

    def draw_detection_boxes(self, results):
        """Draw detection boxes on test canvas"""
        if not hasattr(self, 'test_image'):
            return

        # Redisplay the image first
        self.display_test_image()

        # Get canvas size and scaling info
        canvas_width = self.test_canvas.winfo_width()
        canvas_height = self.test_canvas.winfo_height()
        img_width, img_height = self.test_image.size

        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y)

        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        # Draw boxes
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = box.tolist()

            # Scale coordinates
            scaled_x1 = offset_x + int(x1 * scale)
            scaled_y1 = offset_y + int(y1 * scale)
            scaled_x2 = offset_x + int(x2 * scale)
            scaled_y2 = offset_y + int(y2 * scale)

            # Draw rectangle
            self.test_canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                                           outline='red', width=3)

            # Draw label
            confidence = conf.item()
            class_id = int(cls.item())
            label = f"{class_id}: {confidence:.2f}"

            # Draw label background
            self.test_canvas.create_rectangle(scaled_x1, scaled_y1-25, scaled_x1+len(label)*8, scaled_y1,
                                           fill='red', outline='red')

            # Draw label text
            self.test_canvas.create_text(scaled_x1+2, scaled_y1-12, text=label,
                                       anchor=tk.W, fill='white', font=('Arial', 10, 'bold'))

    def _train_model(self, dataset_path, model_name):
        """Background training function"""
        try:
            # Create output directory
            model_base_name = model_name.replace('.pt', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"training_{timestamp}"

            # Load model
            self.log_training(f"🔄 Loading model: {model_name}")
            model = YOLO(model_name)

            # Training parameters
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            img_size = self.img_size_var.get()

            self.log_training(f"🚀 Starting training with {epochs} epochs, batch size {batch_size}")

            # Train model
            results = model.train(
                data=os.path.join(dataset_path, "data.yaml"),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=project_name,
                name=f"{model_base_name}_trained",
                exist_ok=True,
                verbose=True
            )

            if not self.training_active:
                return

            self.log_training("✅ Training completed successfully!")
            self.training_status.set("Training completed")

            # Save model path for testing
            self.trained_model_path = os.path.join(project_name, f"{model_base_name}_trained", "weights", "best.pt")

        except Exception as e:
            self.log_training(f"❌ Training failed: {str(e)}")
            self.training_status.set("Training failed")

        finally:
            self.training_active = False
            self.training_progress.set(0)

    def log_training(self, message):
        """Log training message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.training_log.see(tk.END)

    def setup_left_panel(self, parent): messagebox, scrolledtext
import os
import json
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from datetime import datetime
import shutil
import sys
import threading
import time
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import subprocess
import platform

class DatasetCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Dataset Creator & Annotator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize training variables
        self.training_thread = None
        self.training_active = False
        self.current_model = None
        self.test_results = []
        self.training_progress = tk.DoubleVar()
        self.training_status = tk.StringVar(value="Ready")

        # Model configurations
        self.available_models = {
            "YOLOv8 Nano": "yolov8n.pt",
            "YOLOv8 Small": "yolov8s.pt",
            "YOLOv8 Medium": "yolov8m.pt",
            "YOLOv8 Large": "yolov8l.pt",
            "YOLOv8 Extra Large": "yolov8x.pt",
            "YOLOv11 Nano": "yolo11n.pt"
        }

        # Colors
        self.colors = {
            'bg': '#f0f0f0',
            'fg': '#333333',
            'accent': '#007acc',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'button': '#007acc',
            'button_hover': '#005999'
        }

        self.setup_ui()
        self.load_existing_datasets()

    def setup_ui(self):
        """Setup the main user interface"""
        # Create main window with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.dataset_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.testing_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.dataset_tab, text="📁 Dataset Creator")
        self.notebook.add(self.training_tab, text="🚀 Model Training")
        self.notebook.add(self.testing_tab, text="🔍 Model Testing")

        # Setup each tab
        self.setup_dataset_tab()
        self.setup_training_tab()
        self.setup_testing_tab()

    def setup_dataset_tab(self):
        """Setup the dataset creation tab"""
        # Create main frame for dataset tab
        main_frame = ttk.Frame(self.dataset_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left panel (controls)
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Create right panel (image display)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)

    def setup_training_tab(self):
        """Setup the model training tab"""
        # Training configuration section
        config_frame = ttk.LabelFrame(self.training_tab, text="⚙️ Training Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(10, 5), padx=10)

        # Model selection
        ttk.Label(config_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_var,
                                  values=list(self.available_models.keys()), state="readonly")
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        model_combo.set("YOLOv8 Nano")

        # Training parameters
        ttk.Label(config_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(config_frame, from_=1, to=500, textvariable=self.epochs_var).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))

        ttk.Label(config_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.batch_size_var).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))

        ttk.Label(config_frame, text="Image Size:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Spinbox(config_frame, from_=320, to=1280, increment=64, textvariable=self.img_size_var).grid(row=3, column=1, sticky=tk.W, pady=2, padx=(10, 0))

        # Dataset selection for training
        ttk.Label(config_frame, text="Dataset:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.train_dataset_var = tk.StringVar()
        self.train_dataset_combo = ttk.Combobox(config_frame, textvariable=self.train_dataset_var, state="readonly")
        self.train_dataset_combo.grid(row=4, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        self.update_training_datasets()

        # Training control section
        control_frame = ttk.LabelFrame(self.training_tab, text="🎮 Training Control", padding=10)
        control_frame.pack(fill=tk.X, pady=5, padx=10)

        # Progress bar
        ttk.Label(control_frame, text="Progress:").pack(anchor=tk.W)
        self.training_progress_bar = ttk.Progressbar(control_frame, variable=self.training_progress, maximum=100)
        self.training_progress_bar.pack(fill=tk.X, pady=(0, 5))

        # Status label
        self.training_status_label = ttk.Label(control_frame, textvariable=self.training_status)
        self.training_status_label.pack(anchor=tk.W, pady=(0, 10))

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="▶️ Start Training", command=self.start_training).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="⏹️ Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="📊 View Results", command=self.view_training_results).pack(side=tk.LEFT)

        # Training log section
        log_frame = ttk.LabelFrame(self.training_tab, text="📋 Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.training_log = scrolledtext.ScrolledText(log_frame, height=15)
        self.training_log.pack(fill=tk.BOTH, expand=True)

    def setup_testing_tab(self):
        """Setup the model testing tab"""
        # Model selection section
        model_frame = ttk.LabelFrame(self.testing_tab, text="🤖 Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(10, 5), padx=10)

        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.test_model_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.test_model_path).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Button(model_frame, text="📁 Browse", command=self.browse_model).grid(row=0, column=2, padx=(5, 0))

        # Test image section
        image_frame = ttk.LabelFrame(self.testing_tab, text="🖼️ Test Image", padding=10)
        image_frame.pack(fill=tk.X, pady=5, padx=10)

        ttk.Button(image_frame, text="📤 Load Test Image", command=self.load_test_image).pack(anchor=tk.W)
        ttk.Button(image_frame, text="🔍 Run Detection", command=self.run_detection).pack(anchor=tk.W, pady=(5, 0))

        # Results section
        results_frame = ttk.LabelFrame(self.testing_tab, text="📊 Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Test image canvas
        self.test_canvas_frame = ttk.Frame(self.testing_tab)
        self.test_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.test_canvas = tk.Canvas(self.test_canvas_frame, bg='gray')
        self.test_canvas.pack(fill=tk.BOTH, expand=True)

    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # Dataset Management Section
        dataset_frame = ttk.LabelFrame(parent, text="📁 Dataset Management", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        # Dataset selection
        ttk.Label(dataset_frame, text="Current Dataset:").pack(anchor=tk.W)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                        state="readonly")
        self.dataset_combo.pack(fill=tk.X, pady=(0, 5))
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)

        # Dataset buttons
        btn_frame = ttk.Frame(dataset_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="📁 Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="➕ New Dataset", command=self.create_new_dataset).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="🗑️ Delete Dataset", command=self.delete_dataset).pack(side=tk.LEFT, padx=(5, 0))

        # Image Management Section
        image_frame = ttk.LabelFrame(parent, text="🖼️ Image Management", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(image_frame, text="📤 Add Images", command=self.add_images).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(image_frame, text="📋 View Images", command=self.view_images).pack(fill=tk.X)

        # Image list
        self.image_listbox = tk.Listbox(image_frame, height=8)
        self.image_listbox.pack(fill=tk.X, pady=(5, 0))
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_selected)

        # Annotation Section
        annotation_frame = ttk.LabelFrame(parent, text="🎯 Annotation Tools", padding=10)
        annotation_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(annotation_frame, text="✅ Save Annotations", command=self.save_annotations).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(annotation_frame, text="↩️ Undo Last", command=self.undo_last_annotation).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(annotation_frame, text="🗑️ Clear All", command=self.clear_all_annotations).pack(fill=tk.X)

        # Annotation list
        ttk.Label(annotation_frame, text="Current Annotations:").pack(anchor=tk.W, pady=(5, 0))
        self.annotation_text = scrolledtext.ScrolledText(annotation_frame, height=6, width=30)
        self.annotation_text.pack(fill=tk.X)

        # Status Section
        status_frame = ttk.LabelFrame(parent, text="📊 Status", padding=10)
        status_frame.pack(fill=tk.X)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=30)
        self.status_text.pack(fill=tk.X)
        self.status_text.insert(tk.END, "Welcome to YOLO Dataset Creator!\n\n")
        self.status_text.insert(tk.END, "1. Create or load a dataset\n")
        self.status_text.insert(tk.END, "2. Add images to your dataset\n")
        self.status_text.insert(tk.END, "3. Select an image to annotate\n")
        self.status_text.insert(tk.END, "4. Draw bounding boxes\n")
        self.status_text.insert(tk.END, "5. Save your annotations\n")
        self.status_text.config(state=tk.DISABLED)

    def setup_right_panel(self, parent):
        """Setup the right image display panel"""
        # Image display area
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for image display
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Instructions
        self.instruction_label = ttk.Label(parent,
                                         text="👆 Click and drag on the image to draw bounding boxes",
                                         foreground=self.colors['accent'])
        self.instruction_label.pack(pady=5)

    def load_existing_datasets(self):
        """Load existing datasets from the datasets folder"""
        if not os.path.exists("datasets"):
            os.makedirs("datasets")
            return

        datasets = []
        for item in os.listdir("datasets"):
            dataset_path = os.path.join("datasets", item)
            if os.path.isdir(dataset_path):
                datasets.append(item)

        self.dataset_combo['values'] = datasets
        if datasets:
            self.dataset_combo.set(datasets[0])
            self.on_dataset_selected(None)

    def create_new_dataset(self):
        """Create a new dataset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Dataset")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Dataset Name:").pack(pady=10)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var)
        name_entry.pack(fill=tk.X, padx=20)
        name_entry.focus()

        def create_dataset():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a dataset name")
                return

            # Validate name
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in name for char in invalid_chars):
                messagebox.showerror("Error", "Dataset name contains invalid characters")
                return

            dataset_path = os.path.join("datasets", name)
            if os.path.exists(dataset_path):
                messagebox.showerror("Error", "Dataset already exists")
                return

            try:
                os.makedirs(dataset_path)
                os.makedirs(os.path.join(dataset_path, "images"))
                os.makedirs(os.path.join(dataset_path, "labels"))

                # Create data.yaml
                data_yaml = f"""# YOLO Dataset Configuration
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

train: images
val: images
test: images

nc: 1
names: ['object']
"""

                with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
                    f.write(data_yaml)

                # Create metadata
                metadata = {
                    "name": name,
                    "created": datetime.now().isoformat(),
                    "total_images": 0,
                    "annotated_images": 0
                }

                with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                self.load_existing_datasets()
                self.dataset_combo.set(name)
                self.on_dataset_selected(None)
                self.log_status(f"✅ Created dataset: {name}")
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to create dataset: {str(e)}")

        ttk.Button(dialog, text="Create", command=create_dataset).pack(pady=10)

        dialog.bind('<Return>', lambda e: create_dataset())

    def load_dataset(self):
        """Load the selected dataset"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return

        self.current_dataset = dataset_name
        self.load_images()
        self.log_status(f"📁 Loaded dataset: {dataset_name}")

    def delete_dataset(self):
        """Delete the selected dataset"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return

        if messagebox.askyesno("Confirm Delete",
                              f"Are you sure you want to delete the dataset '{dataset_name}'?\nThis action cannot be undone!"):
            try:
                dataset_path = os.path.join("datasets", dataset_name)
                shutil.rmtree(dataset_path)
                self.load_existing_datasets()
                self.current_dataset = None
                self.clear_canvas()
                self.log_status(f"🗑️ Deleted dataset: {dataset_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete dataset: {str(e)}")

    def add_images(self):
        """Add images to the current dataset"""
        if not self.current_dataset:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]

        filenames = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=filetypes
        )

        if not filenames:
            return

        dataset_path = os.path.join("datasets", self.current_dataset)
        images_path = os.path.join(dataset_path, "images")

        added_count = 0
        for filename in filenames:
            try:
                # Validate image
                with Image.open(filename) as img:
                    img.verify()

                # Copy image
                base_name = os.path.basename(filename)
                dest_path = os.path.join(images_path, base_name)

                if os.path.exists(dest_path):
                    # Handle duplicate names
                    name, ext = os.path.splitext(base_name)
                    counter = 1
                    while os.path.exists(os.path.join(images_path, f"{name}_{counter}{ext}")):
                        counter += 1
                    dest_path = os.path.join(images_path, f"{name}_{counter}{ext}")

                shutil.copy2(filename, dest_path)
                added_count += 1

            except Exception as e:
                self.log_status(f"❌ Failed to add {os.path.basename(filename)}: {str(e)}")

        if added_count > 0:
            self.load_images()
            self.log_status(f"✅ Added {added_count} image(s) to dataset")

    def load_images(self):
        """Load images from the current dataset"""
        if not self.current_dataset:
            return

        dataset_path = os.path.join("datasets", self.current_dataset)
        images_path = os.path.join(dataset_path, "images")

        if not os.path.exists(images_path):
            return

        self.image_listbox.delete(0, tk.END)

        image_files = []
        for filename in os.listdir(images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                image_files.append(filename)

        image_files.sort()

        for filename in image_files:
            self.image_listbox.insert(tk.END, filename)

    def view_images(self):
        """Refresh the image list"""
        self.load_images()
        self.log_status("📋 Refreshed image list")

    def on_dataset_selected(self, event):
        """Handle dataset selection"""
        dataset_name = self.dataset_var.get()
        if dataset_name:
            self.current_dataset = dataset_name
            self.load_images()
            self.clear_canvas()

    def on_image_selected(self, event):
        """Handle image selection"""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        image_name = self.image_listbox.get(selection[0])
        self.load_image(image_name)

    def load_image(self, image_name):
        """Load and display an image"""
        if not self.current_dataset:
            return

        dataset_path = os.path.join("datasets", self.current_dataset)
        images_path = os.path.join(dataset_path, "images")
        image_path = os.path.join(images_path, image_name)

        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Image file not found")
            return

        try:
            self.current_image_path = image_path
            self.current_image = Image.open(image_path)

            # Load existing annotations
            self.load_annotations(image_name)

            # Display image
            self.display_image()

            self.log_status(f"🖼️ Loaded image: {image_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_annotations(self, image_name):
        """Load existing annotations for the current image"""
        if not self.current_dataset:
            return

        dataset_path = os.path.join("datasets", self.current_dataset)
        labels_path = os.path.join(dataset_path, "labels")
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_path, label_name)

        self.current_annotations = []

        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])

                                # Convert to pixel coordinates
                                if self.current_image:
                                    img_width, img_height = self.current_image.size
                                    x1 = int((x_center - width/2) * img_width)
                                    y1 = int((y_center - height/2) * img_height)
                                    x2 = int((x_center + width/2) * img_width)
                                    y2 = int((y_center + height/2) * img_height)

                                    self.current_annotations.append({
                                        'class_id': class_id,
                                        'bbox': (x1, y1, x2, y2),
                                        'yolo': line
                                    })

                self.update_annotation_display()

            except Exception as e:
                self.log_status(f"❌ Error loading annotations: {str(e)}")

    def display_image(self):
        """Display the current image on the canvas"""
        if not self.current_image:
            return

        # Clear canvas
        self.canvas.delete("all")

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet sized, schedule redraw
            self.root.after(100, self.display_image)
            return

        # Calculate scaling
        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.image_scale = min(scale_x, scale_y)

        # Calculate display size
        display_width = int(img_width * self.image_scale)
        display_height = int(img_height * self.image_scale)

        # Calculate offset to center image
        self.image_offset_x = (canvas_width - display_width) // 2
        self.image_offset_y = (canvas_height - display_height) // 2

        # Resize image
        resized_image = self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)

        # Display image
        self.canvas.create_image(self.image_offset_x, self.image_offset_y,
                               anchor=tk.NW, image=self.photo)

        # Draw existing annotations
        self.draw_annotations()

    def draw_annotations(self):
        """Draw existing annotations on the canvas"""
        for annotation in self.current_annotations:
            x1, y1, x2, y2 = annotation['bbox']

            # Scale coordinates
            scaled_x1 = self.image_offset_x + int(x1 * self.image_scale)
            scaled_y1 = self.image_offset_y + int(y1 * self.image_scale)
            scaled_x2 = self.image_offset_x + int(x2 * self.image_scale)
            scaled_y2 = self.image_offset_y + int(y2 * self.image_scale)

            # Draw rectangle
            self.canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                                       outline='green', width=2)

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if not self.current_image:
            return

        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.image_offset_x) / self.image_scale
        img_y = (event.y - self.image_offset_y) / self.image_scale

        # Check if click is within image bounds
        if self.current_image:
            img_width, img_height = self.current_image.size
            if 0 <= img_x <= img_width and 0 <= img_y <= img_height:
                self.drawing = True
                self.start_x = int(img_x)
                self.start_y = int(img_y)

                # Remove previous rectangle if exists
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
        else:
            self.log_status("❌ No image loaded")

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.drawing or not self.current_image:
            return

        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.image_offset_x) / self.image_scale
        img_y = (event.y - self.image_offset_y) / self.image_scale

        # Remove previous rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)

        # Calculate rectangle coordinates
        if self.start_x is not None and self.start_y is not None:
            x1 = min(self.start_x, int(img_x))
            y1 = min(self.start_y, int(img_y))
            x2 = max(self.start_x, int(img_x))
            y2 = max(self.start_y, int(img_y))

            # Scale to canvas coordinates
            scaled_x1 = self.image_offset_x + int(x1 * self.image_scale)
            scaled_y1 = self.image_offset_y + int(y1 * self.image_scale)
            scaled_x2 = self.image_offset_x + int(x2 * self.image_scale)
            scaled_y2 = self.image_offset_y + int(y2 * self.image_scale)

            # Draw rectangle
            self.rect_id = self.canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                                                      outline='red', width=2, dash=(5, 5))

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.drawing or not self.current_image:
            return

        self.drawing = False

        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.image_offset_x) / self.image_scale
        img_y = (event.y - self.image_offset_y) / self.image_scale

        # Calculate rectangle coordinates
        if self.start_x is not None and self.start_y is not None:
            x1 = min(self.start_x, int(img_x))
            y1 = min(self.start_y, int(img_y))
            x2 = max(self.start_x, int(img_x))
            y2 = max(self.start_y, int(img_y))

            # Check if rectangle is large enough
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.log_status("⚠️ Bounding box too small - try again")
                return

            # Convert to YOLO format
            img_width, img_height = self.current_image.size
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            # Add annotation
            self.current_annotations.append({
                'class_id': 0,
                'bbox': (x1, y1, x2, y2),
                'yolo': yolo_line
            })

            # Remove temporary rectangle
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None

            # Draw permanent rectangle
            scaled_x1 = self.image_offset_x + int(x1 * self.image_scale)
            scaled_y1 = self.image_offset_y + int(y1 * self.image_scale)
            scaled_x2 = self.image_offset_x + int(x2 * self.image_scale)
            scaled_y2 = self.image_offset_y + int(y2 * self.image_scale)

            self.canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                                       outline='green', width=2)

            # Update display
            self.update_annotation_display()
            self.log_status(f"✅ Added bounding box ({{len(self.current_annotations)}} total)")

    def update_annotation_display(self):
        """Update the annotation display"""
        self.annotation_text.config(state=tk.NORMAL)
        self.annotation_text.delete(1.0, tk.END)

        for i, annotation in enumerate(self.current_annotations, 1):
            self.annotation_text.insert(tk.END, f"{i}. {annotation['yolo']}\n")

        self.annotation_text.config(state=tk.DISABLED)

    def save_annotations(self):
        """Save current annotations to file"""
        if not self.current_dataset or not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if not self.current_annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return

        try:
            dataset_path = os.path.join("datasets", self.current_dataset)
            labels_path = os.path.join(dataset_path, "labels")

            if not os.path.exists(labels_path):
                os.makedirs(labels_path)

            image_name = os.path.basename(self.current_image_path)
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(labels_path, label_name)

            with open(label_path, 'w') as f:
                for annotation in self.current_annotations:
                    f.write(annotation['yolo'] + '\n')

            self.log_status(f"💾 Saved {len(self.current_annotations)} annotation(s)")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def undo_last_annotation(self):
        """Remove the last annotation"""
        if not self.current_annotations:
            messagebox.showwarning("Warning", "No annotations to undo")
            return

        self.current_annotations.pop()
        self.display_image()  # Redraw image and annotations
        self.update_annotation_display()
        self.log_status(f"↩️ Removed last annotation ({len(self.current_annotations)} remaining)")

    def clear_all_annotations(self):
        """Clear all annotations for the current image"""
        if not self.current_annotations:
            messagebox.showwarning("Warning", "No annotations to clear")
            return

        if messagebox.askyesno("Confirm Clear",
                              "Are you sure you want to clear all annotations for this image?"):
            self.current_annotations = []
            self.display_image()  # Redraw image without annotations
            self.update_annotation_display()
            self.log_status("🗑️ Cleared all annotations")

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.current_image = None
        self.current_image_path = None
        self.current_annotations = []
        self.update_annotation_display()

    def log_status(self, message):
        """Log a message to the status area"""
        self.status_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

def main():
    """Main function"""
    root = tk.Tk()
    app = DatasetCreator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
