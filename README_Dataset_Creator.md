# YOLO Dataset Creator

A user-friendly GUI application for creating YOLO datasets and annotating images with bounding boxes.

## Features

- **Create Datasets**: Easily create new YOLO-compatible datasets
- **Add Images**: Upload images to your datasets with validation
- **Annotate Images**: Draw bounding boxes directly on images
- **Save Annotations**: Automatically saves in YOLO format
- **View Annotations**: See existing annotations loaded from files
- **Manage Datasets**: Load, delete, and organize your datasets
- **Compatible**: Creates datasets that work seamlessly with the main YOLO Trainer app

## Requirements

- Python 3.6+
- PIL (Pillow)
- Tkinter (usually included with Python)
- NumPy
- OpenCV (cv2)

## Installation

1. Make sure you have all required packages:
```bash
pip install pillow numpy opencv-python
```

2. Tkinter should be included with your Python installation. If not, install it:
   - Windows: Usually included
   - Linux: `sudo apt-get install python3-tk`
   - macOS: Usually included

## Usage

1. **Run the application**:
```bash
python dataset_creator.py
```

2. **Create a new dataset**:
   - Click "New Dataset"
   - Enter a dataset name
   - Click "Create"

3. **Add images**:
   - Click "Add Images"
   - Select image files (JPG, PNG, BMP, TIFF)
   - Images will be copied to your dataset

4. **Annotate images**:
   - Select an image from the list
   - Click and drag on the image to draw bounding boxes
   - Boxes will appear in red while drawing, then turn green when saved

5. **Save annotations**:
   - Click "Save Annotations" to save current boxes
   - Or they will be saved automatically when you select a different image

6. **Manage annotations**:
   - "Undo Last": Remove the most recent bounding box
   - "Clear All": Remove all bounding boxes for current image
   - View current annotations in the text area

## Dataset Structure

The application creates datasets with this structure:
```
datasets/
└── your_dataset_name/
    ├── images/          # Your images
    │   ├── image1.jpg
    │   └── image2.png
    ├── labels/          # YOLO annotation files
    │   ├── image1.txt
    │   └── image2.txt
    ├── data.yaml       # YOLO dataset configuration
    └── metadata.json   # Dataset information
```

## Annotation Format

Annotations are saved in YOLO format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: Object class (currently fixed to 0)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

## Using with YOLO Trainer

1. Create and annotate your dataset using this tool
2. Open the main YOLO Trainer app (`python app.py`)
3. Go to the "Browse" tab
4. Select your dataset from the dropdown
5. Your annotated images will be loaded and ready for training

## Tips

- **Image Quality**: Use high-resolution images for better training results
- **Bounding Boxes**: Draw boxes tightly around objects
- **Consistency**: Be consistent with your annotation style
- **Validation**: The app validates images before adding them
- **Backup**: Keep backups of important datasets

## Troubleshooting

- **Tkinter not found**: Install tkinter for your Python version
- **Images not loading**: Check that images are valid and not corrupted
- **Annotations not saving**: Make sure you have write permissions in the datasets folder
- **Canvas not responding**: Try resizing the window to refresh the canvas

## Keyboard Shortcuts

- **Ctrl+N**: Create new dataset
- **Ctrl+O**: Load dataset
- **Ctrl+S**: Save annotations
- **Ctrl+Z**: Undo last annotation
- **Delete**: Clear all annotations

## License

This tool is provided as-is for creating YOLO datasets.</content>
<parameter name="filePath">f:\Programming\Yolo_Trainer\README_Dataset_Creator.md
