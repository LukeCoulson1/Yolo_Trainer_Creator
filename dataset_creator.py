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
import shutil
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

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

        # Initialize drawing variables
        self.drawing = False
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.image_scale = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0

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

        # Quick mode for fast annotation
        self.quick_mode = tk.BooleanVar(value=False)
        
        # Preset box dimensions (normalized 0-1)
        self.preset_width = tk.DoubleVar(value=0.225)
        self.preset_height = tk.DoubleVar(value=0.143)

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
        ttk.Label(config_frame, text="YOLO model architecture (Nano = fastest, XL = most accurate)", 
                 foreground='gray', font=('', 8)).grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        model_combo.set("YOLOv8 Nano")

        # Training parameters
        ttk.Label(config_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(config_frame, from_=1, to=500, textvariable=self.epochs_var).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Label(config_frame, text="Number of training iterations (more = better but slower)", 
                 foreground='gray', font=('', 8)).grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10, 0))

        ttk.Label(config_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.batch_size_var).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Label(config_frame, text="Images processed simultaneously (higher = faster but needs more VRAM)", 
                 foreground='gray', font=('', 8)).grid(row=2, column=2, sticky=tk.W, pady=2, padx=(10, 0))

        ttk.Label(config_frame, text="Image Size:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Spinbox(config_frame, from_=320, to=1280, increment=64, textvariable=self.img_size_var).grid(row=3, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Label(config_frame, text="Input resolution (higher = more detail but slower & needs more memory)", 
                 foreground='gray', font=('', 8)).grid(row=3, column=2, sticky=tk.W, pady=2, padx=(10, 0))

        # Dataset selection for training
        ttk.Label(config_frame, text="Dataset:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.train_dataset_var = tk.StringVar()
        self.train_dataset_combo = ttk.Combobox(config_frame, textvariable=self.train_dataset_var, state="readonly")
        self.train_dataset_combo.grid(row=4, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Label(config_frame, text="Select dataset with images and annotations for training", 
                 foreground='gray', font=('', 8)).grid(row=4, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.update_training_datasets()

        # Base model selection (for fine-tuning)
        ttk.Label(config_frame, text="Base Model:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.base_model_var = tk.StringVar(value="None (Train from scratch)")
        self.base_model_combo = ttk.Combobox(config_frame, textvariable=self.base_model_var, state="readonly")
        self.base_model_combo.grid(row=5, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Label(config_frame, text="Optional: Start training from a previously trained model", 
                 foreground='gray', font=('', 8)).grid(row=5, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.update_base_models()

        # Model name input
        ttk.Label(config_frame, text="Model Name:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.model_name_var = tk.StringVar()
        model_name_frame = ttk.Frame(config_frame)
        model_name_frame.grid(row=6, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        model_name_entry = ttk.Entry(model_name_frame, textvariable=self.model_name_var, width=25)
        model_name_entry.pack(side=tk.LEFT)
        
        clear_button = ttk.Button(model_name_frame, text="🗑️", width=3, 
                                command=lambda: self.model_name_var.set(""))
        clear_button.pack(side=tk.LEFT, padx=(2, 0))
        
        ttk.Label(config_frame, text="Optional: Custom name for your trained model (auto-generated if empty)", 
                 foreground='gray', font=('', 8)).grid(row=6, column=2, sticky=tk.W, pady=2, padx=(10, 0))

        # Configure column weights for proper layout
        config_frame.columnconfigure(0, weight=0)  # Label column - fixed width
        config_frame.columnconfigure(1, weight=0)  # Control column - fixed width  
        config_frame.columnconfigure(2, weight=1)  # Tips column - expandable

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
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="▶️ Start Training", command=self.start_training).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="⏹️ Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="� Refresh Models", command=self.refresh_models).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="�📊 View Results", command=self.view_training_results).pack(side=tk.LEFT)
        
        # Button tips
        ttk.Label(control_frame, text="💡 Start: Begin training with selected parameters | Stop: Halt current training | Results: View training metrics and charts", 
                 foreground='gray', font=('', 8)).pack(anchor=tk.W, pady=(5, 0))

        # Training log section
        log_frame = ttk.LabelFrame(self.training_tab, text="📋 Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        ttk.Label(log_frame, text="Real-time training progress, metrics, and status messages appear here", 
                 foreground='gray', font=('', 8)).pack(anchor=tk.W, pady=(0, 5))
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
        ttk.Button(image_frame, text="� Add Folder of Images", command=self.add_images_from_folder).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(image_frame, text="🗑️ Remove Image", command=self.remove_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(image_frame, text="�📋 View Images", command=self.view_images).pack(fill=tk.X)

        # Image list
        self.image_listbox = tk.Listbox(image_frame, height=8)
        self.image_listbox.pack(fill=tk.X, pady=(5, 0))
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_selected)
        self.image_listbox.bind('<Delete>', lambda e: self.remove_image())
        self.image_listbox.focus_set()  # Ensure listbox can receive keyboard focus

        # Annotation Section
        annotation_frame = ttk.LabelFrame(parent, text="🎯 Annotation Tools", padding=10)
        annotation_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(annotation_frame, text="✅ Save Annotations", command=self.save_annotations).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(annotation_frame, text="↩️ Undo Last", command=self.undo_last_annotation).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(annotation_frame, text="🗑️ Clear All", command=self.clear_all_annotations).pack(fill=tk.X, pady=(0, 5))
        
        # Quick Mode toggle
        ttk.Checkbutton(annotation_frame, text="⚡ Quick Mode (auto-save & next)", 
                        variable=self.quick_mode).pack(fill=tk.X, pady=(0, 5))
        
        # Preset Box Size Controls
        ttk.Label(annotation_frame, text="📏 Preset Box Size:").pack(anchor=tk.W, pady=(5, 0))
        
        # Width slider
        width_frame = ttk.Frame(annotation_frame)
        width_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(width_frame, text="Width:").pack(side=tk.LEFT)
        ttk.Label(width_frame, textvariable=self.preset_width, width=6).pack(side=tk.RIGHT)
        ttk.Scale(width_frame, from_=0.05, to=0.5, variable=self.preset_width, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Height slider
        height_frame = ttk.Frame(annotation_frame)
        height_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(height_frame, text="Height:").pack(side=tk.LEFT)
        ttk.Label(height_frame, textvariable=self.preset_height, width=6).pack(side=tk.RIGHT)
        ttk.Scale(height_frame, from_=0.05, to=0.5, variable=self.preset_height, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

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
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        # Bind keyboard shortcuts
        self.root.bind('<Control-q>', lambda e: self.quick_mode.set(not self.quick_mode.get()))
        self.root.focus_set()  # Allow root to receive keyboard events

        # Instructions
        self.instruction_label = ttk.Label(parent,
                                         text="👆 Click and drag to draw bounding boxes, or double-click for preset box\n💡 Use sliders to adjust preset box size • Ctrl+Q to toggle Quick Mode",
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
        
        # Also update training datasets
        self.update_training_datasets()

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
                self.update_training_datasets()
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

    def add_images_from_folder(self):
        """Add all images from a selected folder to the current dataset"""
        if not self.current_dataset:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Images"
        )

        if not folder_path:
            return

        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        # Find all image files in the folder (recursive)
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)

        if not image_files:
            messagebox.showinfo("No Images Found", f"No image files found in {folder_path}")
            return

        dataset_path = os.path.join("datasets", self.current_dataset)
        images_path = os.path.join(dataset_path, "images")

        added_count = 0
        skipped_count = 0

        for image_path in image_files:
            try:
                # Validate image
                with Image.open(image_path) as img:
                    img.verify()

                # Copy image
                base_name = os.path.basename(image_path)
                dest_path = os.path.join(images_path, base_name)

                if os.path.exists(dest_path):
                    # Handle duplicate names
                    name, ext = os.path.splitext(base_name)
                    counter = 1
                    while os.path.exists(os.path.join(images_path, f"{name}_{counter}{ext}")):
                        counter += 1
                    dest_path = os.path.join(images_path, f"{name}_{counter}{ext}")

                shutil.copy2(image_path, dest_path)
                added_count += 1

            except Exception as e:
                self.log_status(f"❌ Failed to add {os.path.basename(image_path)}: {str(e)}")
                skipped_count += 1

        if added_count > 0:
            self.load_images()
            status_msg = f"✅ Added {added_count} image(s) from folder"
            if skipped_count > 0:
                status_msg += f" ({skipped_count} skipped due to errors)"
            self.log_status(status_msg)
        else:
            messagebox.showwarning("Warning", "No images could be added from the selected folder")

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

    def remove_image(self):
        """Remove the selected image from the dataset"""
        if not self.current_dataset:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        # Get selected image
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an image to remove")
            return

        current_index = selection[0]
        image_name = self.image_listbox.get(current_index)

        try:
            dataset_path = os.path.join("datasets", self.current_dataset)
            images_path = os.path.join(dataset_path, "images")
            labels_path = os.path.join(dataset_path, "labels")

            # Remove image file
            image_path = os.path.join(images_path, image_name)
            if os.path.exists(image_path):
                os.remove(image_path)

            # Remove corresponding label file
            name, ext = os.path.splitext(image_name)
            label_path = os.path.join(labels_path, f"{name}.txt")
            if os.path.exists(label_path):
                os.remove(label_path)

            # Clear canvas if this image was currently displayed
            if hasattr(self, 'current_image_path') and self.current_image_path:
                current_image_name = os.path.basename(self.current_image_path)
                if current_image_name == image_name:
                    self.clear_canvas()

            # Reload images
            self.load_images()

            # Select next image automatically
            self._select_next_image_after_removal(current_index)

            self.log_status(f"🗑️ Removed image: {image_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove image: {str(e)}")

    def _select_next_image_after_removal(self, removed_index):
        """Select the next appropriate image after removal"""
        if self.image_listbox.size() == 0:
            # No images left
            self.clear_canvas()
            self.log_status("📭 No images remaining in dataset")
            return

        # Try to select the image at the same index (now contains the next image)
        if removed_index < self.image_listbox.size():
            next_index = removed_index
        else:
            # If we removed the last image, select the new last image
            next_index = self.image_listbox.size() - 1

        # Select and load the next image
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(next_index)
        self.image_listbox.see(next_index)
        image_name = self.image_listbox.get(next_index)
        self.load_image(image_name)
        self.log_status(f"📸 Auto-selected next image: {image_name}")

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

    def next_image(self):
        """Move to the next image in the list"""
        if not self.current_dataset:
            return
            
        # Get current selection
        selection = self.image_listbox.curselection()
        if not selection:
            # No selection, select first image
            if self.image_listbox.size() > 0:
                self.image_listbox.selection_set(0)
                self.image_listbox.see(0)
                image_name = self.image_listbox.get(0)
                self.load_image(image_name)
            return
            
        current_index = selection[0]
        next_index = current_index + 1
        
        # Check if we're at the end
        if next_index >= self.image_listbox.size():
            self.log_status("📸 Reached end of image list")
            return
            
        # Select next image
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(next_index)
        self.image_listbox.see(next_index)
        image_name = self.image_listbox.get(next_index)
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
            self.log_status(f"✅ Added bounding box ({len(self.current_annotations)} total)")

    def on_double_click(self, event):
        """Handle double-click to create preset bounding box"""
        if not self.current_image:
            return

        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.image_offset_x) / self.image_scale
        img_y = (event.y - self.image_offset_y) / self.image_scale

        # Check if click is within image bounds
        if self.current_image:
            img_width, img_height = self.current_image.size
            if 0 <= img_x <= img_width and 0 <= img_y <= img_height:
                # Use adjustable size from sliders, center at clicked location
                class_id = 0  # Fixed: Use class ID 0 (0-based indexing)
                width = self.preset_width.get()   # Get width from slider
                height = self.preset_height.get()  # Get height from slider

                # Calculate center based on clicked position
                x_center = img_x / img_width
                y_center = img_y / img_height

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)

                # Create YOLO format string
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                # Add annotation
                self.current_annotations.append({
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'yolo': yolo_line
                })

                # Draw the rectangle on canvas
                scaled_x1 = self.image_offset_x + int(x1 * self.image_scale)
                scaled_y1 = self.image_offset_y + int(y1 * self.image_scale)
                scaled_x2 = self.image_offset_x + int(x2 * self.image_scale)
                scaled_y2 = self.image_offset_y + int(y2 * self.image_scale)

                self.canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                                           outline='green', width=2)

                # Update display
                self.update_annotation_display()
                self.log_status(f"✅ Added preset bounding box at clicked location ({len(self.current_annotations)} total)")
                
                # Quick mode: auto-save and move to next image
                if self.quick_mode.get():
                    self.save_annotations()
                    self.next_image()
            else:
                self.log_status("❌ Double-click outside image bounds")
        else:
            self.log_status("❌ No image loaded")

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

            if self.current_annotations:
                self.log_status(f"💾 Saved {len(self.current_annotations)} annotation(s)")
            else:
                self.log_status("💾 Saved empty annotations (cleared)")

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
        self.log_status(f"↩️ Removed last annotation ({len(self.current_annotations)} remaining) - remember to save!")

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
            self.log_status("🗑️ Cleared all annotations (remember to save!)")

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

    def update_training_datasets(self):
        """Update the list of available datasets for training"""
        if not os.path.exists("datasets"):
            self.train_dataset_combo['values'] = []
            return

        datasets = []
        for item in os.listdir("datasets"):
            dataset_path = os.path.join("datasets", item)
            if os.path.isdir(dataset_path):
                # Check if dataset has required files
                data_yaml = os.path.join(dataset_path, "data.yaml")
                images_path = os.path.join(dataset_path, "images")
                if os.path.exists(data_yaml) and os.path.exists(images_path):
                    datasets.append(item)

        self.train_dataset_combo['values'] = datasets
        if datasets:
            self.train_dataset_combo.set(datasets[0])

    def update_base_models(self):
        """Update the list of available base models for fine-tuning"""
        base_models = ["None (Train from scratch)"]
        
        # Add models from the models directory
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith('.pt'):
                    base_models.append(f"models/{file}")
        
        # Add models from recent training sessions
        if os.path.exists("training_*/"):
            import glob
            training_dirs = glob.glob("training_*/")
            for training_dir in sorted(training_dirs, reverse=True)[:5]:  # Last 5 training sessions
                training_name = os.path.basename(training_dir.rstrip('/\\'))
                weights_path = os.path.join(training_dir, "**", "weights", "best.pt")
                if glob.glob(weights_path):
                    base_models.append(f"{training_name}/best.pt")
        
        self.base_model_combo['values'] = base_models
        if not hasattr(self, 'base_model_var') or not self.base_model_var.get():
            self.base_model_var.set("None (Train from scratch)")

    def refresh_models(self):
        """Refresh both dataset and base model lists"""
        self.update_training_datasets()
        self.update_base_models()
        self.log_training("🔄 Refreshed available datasets and models")

    def sanitize_filename(self, filename):
        """Sanitize filename to be safe for filesystem"""
        import re
        
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing whitespace and dots
        filename = filename.strip(' .')
        
        # Ensure it's not empty
        if not filename:
            return f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename

    def validate_dataset_labels(self, dataset_path):
        """Validate that all label files have correct class IDs"""
        try:
            # Read data.yaml to get number of classes
            data_yaml_path = os.path.join(dataset_path, "data.yaml")
            if not os.path.exists(data_yaml_path):
                messagebox.showerror("Error", "data.yaml not found")
                return False

            import yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)

            num_classes = data_config.get('nc', 0)
            if num_classes == 0:
                messagebox.showerror("Error", "No classes defined in data.yaml")
                return False

            # Check all label files
            labels_path = os.path.join(dataset_path, "labels")
            if not os.path.exists(labels_path):
                messagebox.showerror("Error", "Labels directory not found")
                return False

            label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
            if not label_files:
                messagebox.showwarning("Warning", "No label files found. Training with empty labels.")
                return True

            invalid_files = []
            max_class_id = num_classes - 1  # 0-based indexing

            for label_file in label_files:
                label_path = os.path.join(labels_path, label_file)
                try:
                    with open(label_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split()
                            if len(parts) < 5:
                                continue

                            try:
                                class_id = int(parts[0])
                                if class_id < 0 or class_id > max_class_id:
                                    invalid_files.append(f"{label_file}:line {line_num} (class {class_id}, valid: 0-{max_class_id})")
                            except ValueError:
                                invalid_files.append(f"{label_file}:line {line_num} (invalid class ID)")

                except Exception as e:
                    invalid_files.append(f"{label_file}: {str(e)}")

            if invalid_files:
                # Show error with option to auto-fix
                error_msg = f"Found {len(invalid_files)} label validation errors:\n\n"
                error_msg += "\n".join(invalid_files[:10])  # Show first 10 errors
                if len(invalid_files) > 10:
                    error_msg += f"\n... and {len(invalid_files) - 10} more errors"

                error_msg += f"\n\nValid class IDs: 0-{max_class_id} (for {num_classes} classes)"
                error_msg += "\n\nWould you like to auto-fix these issues?"

                if messagebox.askyesno("Label Validation Errors", error_msg):
                    return self.auto_fix_labels(labels_path, max_class_id)
                else:
                    return False

            self.log_training(f"✅ Label validation passed - {len(label_files)} files checked")
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Label validation failed: {str(e)}")
            return False

    def auto_fix_labels(self, labels_path, max_class_id):
        """Automatically fix common label issues"""
        try:
            import shutil
            from datetime import datetime

            # Create backup
            backup_dir = os.path.join(os.path.dirname(labels_path), "labels_backup")
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(labels_path, backup_dir)

            self.log_training("📦 Created backup of original labels")

            fixed_count = 0
            label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

            for label_file in label_files:
                label_path = os.path.join(labels_path, label_file)
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    modified = False
                    new_lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                # Fix common issues:
                                if class_id < 0:
                                    parts[0] = '0'  # Negative class IDs -> 0
                                    modified = True
                                elif class_id > max_class_id:
                                    parts[0] = str(max_class_id)  # Too high class IDs -> max valid
                                    modified = True
                            except ValueError:
                                continue  # Skip invalid lines

                        new_lines.append(' '.join(parts))

                    if modified:
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(new_lines) + '\n')
                        fixed_count += 1

                except Exception as e:
                    self.log_training(f"⚠️ Error fixing {label_file}: {str(e)}")

            # Clear any cache files
            cache_files = [f for f in os.listdir(os.path.dirname(labels_path)) if f.endswith('.cache')]
            for cache_file in cache_files:
                cache_path = os.path.join(os.path.dirname(labels_path), cache_file)
                try:
                    os.remove(cache_path)
                    self.log_training(f"🗑️ Removed cache file: {cache_file}")
                except:
                    pass

            self.log_training(f"🔧 Auto-fixed {fixed_count} label files")
            self.log_training(f"📂 Backup saved to: {backup_dir}")

            if fixed_count > 0:
                messagebox.showinfo("Auto-Fix Complete",
                                  f"Fixed {fixed_count} label files.\n"
                                  f"Backup saved to: {os.path.basename(backup_dir)}\n\n"
                                  "You can now proceed with training.")
                return True
            else:
                messagebox.showinfo("No Fixes Needed", "All labels were already valid.")
                return True

        except Exception as e:
            messagebox.showerror("Error", f"Auto-fix failed: {str(e)}")
            return False

    def start_training(self):
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

        # Validate dataset labels before training
        if not self.validate_dataset_labels(dataset_path):
            return

        # Get base model for fine-tuning (if selected)
        base_model = self.base_model_var.get()
        if base_model == "None (Train from scratch)":
            base_model = None
        elif base_model.startswith("models/"):
            base_model = base_model  # Use the model from models directory
        elif "/best.pt" in base_model:
            # Extract the full path for training session models
            training_name = base_model.split('/')[0]
            base_model = os.path.join(training_name, "**", "weights", "best.pt")

        # Preview the model name that will be created
        custom_name = self.model_name_var.get().strip()
        if custom_name:
            preview_name = f"{self.sanitize_filename(custom_name)}.pt"
        else:
            dataset_name = os.path.basename(dataset_path)
            model_base_name = model_name.replace('.pt', '')
            if base_model:
                base_info = "finetuned"
                if "yolov8" in base_model.lower():
                    base_info = f"ft_{os.path.basename(base_model).replace('.pt', '')}"
                elif "training_" in base_model:
                    base_info = f"ft_{base_model.split('/')[0]}"
            else:
                base_info = "scratch"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_name = f"yolo_{dataset_name}_{model_base_name}_{base_info}_{timestamp}.pt"
        
        self.log_training(f"📝 Model will be saved as: {preview_name}")

        # Start training in background thread
        self.training_active = True
        self.training_thread = threading.Thread(target=self._train_model,
                                              args=(dataset_path, model_name, base_model),
                                              daemon=True)
        self.training_thread.start()

        self.log_training("🚀 Starting training...")
        if base_model:
            self.log_training(f"🔄 Fine-tuning from: {base_model}")
        self.training_status.set("Training in progress...")

    def stop_training(self):
        """Stop current training"""
        if self.training_active:
            self.training_active = False
            self.log_training("⏹️ Training stopped by user")
            self.training_status.set("Training stopped")

    def view_training_results(self):
        """View training results in a comprehensive viewer"""
        # Find the most recent training directory
        import glob
        import os

        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)  # Ensure we're in the right directory

        training_dirs = glob.glob("training_*/")
        if not training_dirs:
            # Try with full path
            training_dirs = glob.glob(os.path.join(script_dir, "training_*/"))

        if not training_dirs:
            messagebox.showwarning("No Results", "No training results found. Please run a training session first.")
            return
        
        # Get the most recent training directory (sort by full name for correct chronological order)
        latest_training = max(training_dirs, key=lambda x: x)
        
        # Remove trailing slash if present
        latest_training = latest_training.rstrip('/').rstrip('\\')        # Look for results in the training directory
        results_dir = None
        try:
            for item in os.listdir(latest_training):
                item_path = os.path.join(latest_training, item)
                if os.path.isdir(item_path):
                    # Check for results files
                    try:
                        files_in_dir = os.listdir(item_path)
                        has_results = any(f.lower().endswith(('.csv', '.png', '.jpg')) for f in files_in_dir)
                        if has_results:
                            results_dir = item_path
                            break
                    except (OSError, PermissionError):
                        continue  # Skip directories we can't access
        except (OSError, PermissionError):
            messagebox.showwarning("Error", f"Cannot access training directory: {latest_training}")
            return

        if not results_dir:
            # Try to find results in the training directory itself
            try:
                files_in_training = os.listdir(latest_training)
                has_results = any(f.lower().endswith(('.csv', '.png', '.jpg')) for f in files_in_training)
                if has_results:
                    results_dir = latest_training
                else:
                    messagebox.showwarning("No Results", f"No training results found in {latest_training}")
                    return
            except (OSError, PermissionError):
                messagebox.showwarning("Error", f"Cannot access training directory: {latest_training}")
                return

        # Create results viewer window
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Training Results - {os.path.basename(results_dir)}")
        results_window.geometry("1200x800")

        # Create notebook for different result views
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="📊 Overview")

        # Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="📈 Metrics")

        # Visualizations tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="📉 Charts")

        # Samples tab
        samples_frame = ttk.Frame(notebook)
        notebook.add(samples_frame, text="🖼️ Samples")

        # Load and display results
        self.load_results_overview(overview_frame, results_dir)
        self.load_results_metrics(metrics_frame, results_dir)
        self.load_results_visualizations(viz_frame, results_dir)
        self.load_results_samples(samples_frame, results_dir)

    def load_results_overview(self, parent, results_dir):
        """Load and display training overview"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(scrollable_frame, text="Training Results Overview",
                 font=('', 14, 'bold')).pack(pady=(10, 20))

        # Training info
        info_frame = ttk.LabelFrame(scrollable_frame, text="Training Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        # Load results.csv for metrics
        results_csv = os.path.join(results_dir, "results.csv")
        if os.path.exists(results_csv):
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)

                # Get final metrics
                final_row = df.iloc[-1]

                # Display key metrics
                metrics_data = [
                    ("Total Epochs", len(df)),
                    ("Final mAP@0.5", ".3f"),
                    ("Final mAP@0.5:0.95", ".3f"),
                    ("Final Precision", ".3f"),
                    ("Final Recall", ".3f"),
                    ("Best mAP@0.5", ".3f"),
                ]

                for i, (label, col) in enumerate(metrics_data):
                    if col in df.columns:
                        value = final_row[col]
                        ttk.Label(info_frame, text=f"{label}:").grid(row=i//2, column=i%4, sticky=tk.W, padx=5, pady=2)
                        ttk.Label(info_frame, text=f"{value:.3f}").grid(row=i//2, column=i%4+1, sticky=tk.W, padx=5, pady=2)

            except ImportError:
                ttk.Label(info_frame, text="Install pandas to view detailed metrics: pip install pandas").pack()
            except Exception as e:
                ttk.Label(info_frame, text=f"Error loading metrics: {str(e)}").pack()

        # Model info
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Information", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 20))

        # Load args.yaml for training parameters
        args_yaml = os.path.join(results_dir, "args.yaml")
        if os.path.exists(args_yaml):
            try:
                import yaml
                with open(args_yaml, 'r') as f:
                    args = yaml.safe_load(f)

                # Display key parameters
                params = [
                    ("Model", args.get('model', 'Unknown')),
                    ("Image Size", args.get('imgsz', 'Unknown')),
                    ("Batch Size", args.get('batch', 'Unknown')),
                    ("Epochs", args.get('epochs', 'Unknown')),
                    ("Device", args.get('device', 'Unknown')),
                ]

                for i, (label, value) in enumerate(params):
                    ttk.Label(model_frame, text=f"{label}:").grid(row=i//2, column=i%4, sticky=tk.W, padx=5, pady=2)
                    ttk.Label(model_frame, text=str(value)).grid(row=i//2, column=i%4+1, sticky=tk.W, padx=5, pady=2)

            except Exception as e:
                ttk.Label(model_frame, text=f"Error loading model info: {str(e)}").pack()

        # Weights location
        weights_frame = ttk.LabelFrame(scrollable_frame, text="Trained Weights", padding=10)
        weights_frame.pack(fill=tk.X, pady=(0, 20))

        weights_dir = os.path.join(results_dir, "weights")
        if os.path.exists(weights_dir):
            for weight_file in ["best.pt", "last.pt"]:
                weight_path = os.path.join(weights_dir, weight_file)
                if os.path.exists(weight_path):
                    file_size = os.path.getsize(weight_path) / (1024 * 1024)  # MB
                    ttk.Label(weights_frame, text=f"{weight_file}:").pack(anchor=tk.W)
                    ttk.Label(weights_frame, text=f"  {file_size:.1f} MB").pack(anchor=tk.W, padx=(20, 0))

    def load_results_metrics(self, parent, results_dir):
        """Load and display training metrics"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(scrollable_frame, text="Training Metrics",
                 font=('', 14, 'bold')).pack(pady=(10, 20))

        # Load results.csv
        results_csv = os.path.join(results_dir, "results.csv")
        if os.path.exists(results_csv):
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)

                # Create metrics table
                metrics_frame = ttk.LabelFrame(scrollable_frame, text="Epoch-by-Epoch Metrics", padding=10)
                metrics_frame.pack(fill=tk.BOTH, expand=True)

                # Create treeview for metrics
                columns = list(df.columns)
                tree = ttk.Treeview(metrics_frame, columns=columns, show='headings', height=15)

                # Configure columns
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=80)

                # Add scrollbar
                v_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=tree.yview)
                h_scrollbar = ttk.Scrollbar(metrics_frame, orient="horizontal", command=tree.xview)
                tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

                # Pack tree and scrollbars
                tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

                # Insert data (show last 20 epochs)
                for idx, row in df.tail(20).iterrows():
                    tree.insert("", tk.END, values=[f"{val:.4f}" if isinstance(val, (int, float)) else str(val) for val in row])

            except ImportError:
                ttk.Label(scrollable_frame, text="Install pandas to view metrics table: pip install pandas").pack(pady=20)
            except Exception as e:
                ttk.Label(scrollable_frame, text=f"Error loading metrics: {str(e)}").pack(pady=20)
        else:
            ttk.Label(scrollable_frame, text="No results.csv file found").pack(pady=20)

    def load_results_visualizations(self, parent, results_dir):
        """Load and display training visualizations"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(scrollable_frame, text="Training Visualizations",
                 font=('', 14, 'bold')).pack(pady=(10, 20))

        # Display available charts
        chart_files = [
            ("results.png", "Training Curves"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("confusion_matrix_normalized.png", "Normalized Confusion Matrix"),
            ("BoxF1_curve.png", "F1 Score Curve"),
            ("BoxPR_curve.png", "Precision-Recall Curve"),
            ("BoxP_curve.png", "Precision Curve"),
            ("BoxR_curve.png", "Recall Curve"),
        ]

        for chart_file, title in chart_files:
            chart_path = os.path.join(results_dir, chart_file)
            if os.path.exists(chart_path):
                # Create frame for each chart
                chart_frame = ttk.LabelFrame(scrollable_frame, text=title, padding=10)
                chart_frame.pack(fill=tk.X, pady=(0, 10))

                try:
                    from PIL import Image, ImageTk
                    # Load and display image
                    image = Image.open(chart_path)

                    # Resize if too large
                    max_width = 600
                    if image.width > max_width:
                        ratio = max_width / image.width
                        new_height = int(image.height * ratio)
                        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

                    photo = ImageTk.PhotoImage(image)

                    # Create label to display image
                    image_label = ttk.Label(chart_frame, image=photo)
                    image_label.image = photo  # Keep reference
                    image_label.pack(pady=5)

                except Exception as e:
                    ttk.Label(chart_frame, text=f"Error loading {chart_file}: {str(e)}").pack()

        if not any(os.path.exists(os.path.join(results_dir, chart_file)) for chart_file, _ in chart_files):
            ttk.Label(scrollable_frame, text="No visualization files found").pack(pady=20)

    def load_results_samples(self, parent, results_dir):
        """Load and display training/validation samples"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(scrollable_frame, text="Training & Validation Samples",
                 font=('', 14, 'bold')).pack(pady=(10, 20))

        # Find sample images
        sample_files = []
        for filename in os.listdir(results_dir):
            if filename.startswith(('train_batch', 'val_batch')) and filename.endswith('.jpg'):
                sample_files.append(filename)

        if sample_files:
            # Group by type
            train_samples = [f for f in sample_files if f.startswith('train_batch')]
            val_samples = [f for f in sample_files if f.startswith('val_batch')]

            # Display training samples
            if train_samples:
                train_frame = ttk.LabelFrame(scrollable_frame, text="Training Batch Samples", padding=10)
                train_frame.pack(fill=tk.X, pady=(0, 10))

                for sample_file in sorted(train_samples)[:6]:  # Show first 6
                    self.display_sample_image(train_frame, results_dir, sample_file)

            # Display validation samples
            if val_samples:
                val_frame = ttk.LabelFrame(scrollable_frame, text="Validation Samples", padding=10)
                val_frame.pack(fill=tk.X, pady=(0, 10))

                for sample_file in sorted(val_samples)[:6]:  # Show first 6
                    self.display_sample_image(val_frame, results_dir, sample_file)
        else:
            ttk.Label(scrollable_frame, text="No sample images found").pack(pady=20)

    def display_sample_image(self, parent, results_dir, filename):
        """Display a sample image in the results viewer"""
        try:
            from PIL import Image, ImageTk

            image_path = os.path.join(results_dir, filename)
            image = Image.open(image_path)

            # Resize for display
            max_width = 300
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)

            # Create frame for image and label
            image_frame = ttk.Frame(parent)
            image_frame.pack(side=tk.LEFT, padx=5, pady=5)

            # Image label
            image_label = ttk.Label(image_frame, image=photo)
            image_label.image = photo  # Keep reference
            image_label.pack()

            # Filename label
            ttk.Label(image_frame, text=filename, font=('', 8)).pack()

        except Exception as e:
            ttk.Label(parent, text=f"Error loading {filename}: {str(e)}").pack(side=tk.LEFT, padx=5, pady=5)

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

    def _train_model(self, dataset_path, model_name, base_model=None):
        """Background training function"""
        try:
            # Create output directory
            model_base_name = model_name.replace('.pt', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"training_{timestamp}"

            # Load model (use base model if provided)
            if base_model:
                self.log_training(f"🔄 Loading base model: {base_model}")
                model = YOLO(base_model)
            else:
                self.log_training(f"🔄 Loading model: {model_name}")
                model = YOLO(model_name)

            # Training parameters
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            img_size = self.img_size_var.get()

            training_type = "fine-tuning" if base_model else "training from scratch"
            self.log_training(f"🚀 Starting {training_type} with {epochs} epochs, batch size {batch_size}")

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
                # Removed device='cpu' - now uses GPU automatically
            )

            if not self.training_active:
                return

            self.log_training("✅ Training completed successfully!")
            self.training_status.set("Training completed")

            # Copy the trained model to models directory
            try:
                # Get dataset name from path
                dataset_name = os.path.basename(dataset_path)
                
                # Get custom model name or generate automatic name
                custom_name = self.model_name_var.get().strip()
                if custom_name:
                    # Validate and sanitize custom name
                    custom_name = self.sanitize_filename(custom_name)
                    model_filename = f"{custom_name}.pt"
                else:
                    # Create automatic model filename
                    if base_model:
                        base_info = "finetuned"
                        if "yolov8" in base_model.lower():
                            base_info = f"ft_{os.path.basename(base_model).replace('.pt', '')}"
                        elif "training_" in base_model:
                            base_info = f"ft_{base_model.split('/')[0]}"
                    else:
                        base_info = "scratch"
                    
                    model_filename = f"yolo_{dataset_name}_{model_base_name}_{base_info}_{timestamp}.pt"
                
                models_dir = "models"
                
                # Ensure models directory exists
                os.makedirs(models_dir, exist_ok=True)
                
                # Source path (best.pt from training)
                source_path = os.path.join(project_name, f"{model_base_name}_trained", "weights", "best.pt")
                
                # Destination path in models directory
                dest_path = os.path.join(models_dir, model_filename)
                
                if os.path.exists(source_path):
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    self.log_training(f"💾 Model saved to: {dest_path}")
                    
                    # Update the trained model path for testing
                    self.trained_model_path = dest_path
                    
                    # Refresh available models list
                    self.root.after(1000, self.update_base_models)  # Refresh after 1 second
                else:
                    self.log_training("⚠️ Warning: Could not find best.pt file to copy")
                    # Fallback to original path
                    self.trained_model_path = source_path
                    
            except Exception as copy_error:
                self.log_training(f"⚠️ Warning: Could not copy model to models directory: {str(copy_error)}")
                # Fallback to original path
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


def main():
    """Main function"""
    root = tk.Tk()
    app = DatasetCreator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
