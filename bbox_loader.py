"""
Coordinate Loader Utility
Functions to load bounding box coordinates saved by the Bounding Box Annotator
"""

import json
import os
import pandas as pd
from pathlib import Path

def load_bbox_coordinates(json_file_path):
    """
    Load bounding box coordinates from a JSON file created by bbox_annotator.py

    Args:
        json_file_path (str): Path to the JSON file containing coordinates

    Returns:
        dict: Dictionary containing coordinate data or None if error
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Validate the data structure
        if 'bounding_boxes' not in data:
            print(f"❌ Error: No 'bounding_boxes' key found in {json_file_path}")
            return None

        if not isinstance(data['bounding_boxes'], list):
            print(f"❌ Error: 'bounding_boxes' should be a list in {json_file_path}")
            return None

        print(f"✅ Loaded {len(data['bounding_boxes'])} bounding boxes from {json_file_path}")
        return data

    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {json_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading coordinates: {e}")
        return None

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format

    Args:
        bbox (dict): Bounding box with keys: x, y, width, height
        img_width (int): Image width
        img_height (int): Image height

    Returns:
        str: YOLO format annotation line
    """
    x_center = (bbox['x'] + bbox['width'] / 2) / img_width
    y_center = (bbox['y'] + bbox['height'] / 2) / img_height
    width = bbox['width'] / img_width
    height = bbox['height'] / img_height

    # YOLO format: class x_center y_center width height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def save_yolo_annotations(bboxes, output_path, img_width, img_height):
    """
    Save bounding boxes as YOLO format annotations

    Args:
        bboxes (list): List of bounding box dictionaries
        output_path (str): Path to save the annotation file
        img_width (int): Image width
        img_height (int): Image height
    """
    try:
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                yolo_line = convert_bbox_to_yolo(bbox, img_width, img_height)
                f.write(yolo_line + '\n')

        print(f"✅ Saved {len(bboxes)} annotations to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error saving annotations: {e}")
        return False

def list_bbox_files(directory="."):
    """
    List all JSON files that contain bounding box coordinates

    Args:
        directory (str): Directory to search in

    Returns:
        list: List of JSON files containing bbox data
    """
    bbox_files = []
    path = Path(directory)

    for json_file in path.glob("bboxes_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'bounding_boxes' in data and isinstance(data['bounding_boxes'], list):
                bbox_files.append({
                    'file': str(json_file),
                    'name': json_file.name,
                    'bbox_count': len(data['bounding_boxes']),
                    'image_name': data.get('image_name', 'Unknown'),
                    'created_at': data.get('created_at', 'Unknown')
                })
        except:
            continue

    return bbox_files

def display_bbox_summary(bbox_data):
    """
    Display a summary of bounding box data

    Args:
        bbox_data (dict): Data loaded from load_bbox_coordinates
    """
    if not bbox_data:
        print("❌ No data to display")
        return

    print("\n📊 Bounding Box Summary")
    print("=" * 50)
    print(f"Image: {bbox_data.get('image_name', 'Unknown')}")
    print(f"Image Size: {bbox_data.get('image_size', 'Unknown')}")
    print(f"Total Boxes: {len(bbox_data['bounding_boxes'])}")
    print(f"Created: {bbox_data.get('created_at', 'Unknown')}")

    if bbox_data['bounding_boxes']:
        print("\n📋 Box Details:")
        print("-" * 50)
        print("ID".ljust(5), "X".ljust(8), "Y".ljust(8), "Width".ljust(8), "Height".ljust(8), "Area")
        print("-" * 50)

        for i, bbox in enumerate(bbox_data['bounding_boxes']):
            area = bbox['width'] * bbox['height']
            print(f"{str(i+1).ljust(5)} {str(bbox['x']).ljust(8)} {str(bbox['y']).ljust(8)} {str(bbox['width']).ljust(8)} {str(bbox['height']).ljust(8)} {area}")

if __name__ == "__main__":
    # Example usage
    print("🔧 Bounding Box Coordinate Loader")
    print("Usage: python bbox_loader.py <json_file_path>")

    # List available bbox files
    bbox_files = list_bbox_files()
    if bbox_files:
        print(f"\n📁 Found {len(bbox_files)} bounding box files:")
        for file_info in bbox_files:
            print(f"  • {file_info['name']} ({file_info['bbox_count']} boxes)")
    else:
        print("\n📁 No bounding box files found in current directory")
