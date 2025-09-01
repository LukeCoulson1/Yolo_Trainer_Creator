#!/usr/bin/env python3
"""
Test the training results directory finding logic
"""

import glob
import os

def test_find_training_results():
    """Test the logic for finding training results"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Ensure we're in the right directory

    print(f"Current directory: {os.getcwd()}")

    training_dirs = glob.glob("training_*/")
    if not training_dirs:
        # Try with full path
        training_dirs = glob.glob(os.path.join(script_dir, "training_*/"))

    print(f"Found training directories: {training_dirs}")

    if not training_dirs:
        print("No training directories found!")
        return

    # Get the most recent training directory (sort by full name for correct chronological order)
    latest_training = max(training_dirs, key=lambda x: x)

    # Remove trailing slash if present
    latest_training = latest_training.rstrip('/').rstrip('\\')

    print(f"Selected latest training: {latest_training}")
    print(f"Latest training exists: {os.path.exists(latest_training)}")
    print(f"Latest training is dir: {os.path.isdir(latest_training)}")

    # Look for results in the training directory
    results_dir = None
    try:
        items = os.listdir(latest_training)
        print(f"Items in {latest_training}: {items}")

        for item in items:
            item_path = os.path.join(latest_training, item)
            print(f"Checking {item_path}, is_dir: {os.path.isdir(item_path)}")
            if os.path.isdir(item_path):
                # Check for results files
                try:
                    files_in_dir = os.listdir(item_path)
                    print(f"Files in {item}: {files_in_dir}")
                    has_results = any(f.lower().endswith(('.csv', '.png', '.jpg')) for f in files_in_dir)
                    print(f"{item} has results: {has_results}")
                    if has_results:
                        results_dir = item_path
                        print(f"Found results dir: {results_dir}")
                        break
                except (OSError, PermissionError) as e:
                    print(f"Error accessing {item_path}: {e}")
                    continue  # Skip directories we can't access
    except (OSError, PermissionError) as e:
        print(f"Cannot access training directory: {e}")
        return

    print(f"Final results_dir: {results_dir}")

    if results_dir:
        print(f"✅ Success! Results found in: {results_dir}")
    else:
        print("❌ No results directory found")

if __name__ == "__main__":
    test_find_training_results()
