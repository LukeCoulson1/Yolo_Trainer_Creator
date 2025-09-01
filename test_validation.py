#!/usr/bin/env python3
"""
Test the label validation system
"""

import os
import tempfile
import yaml

def create_test_dataset():
    """Create a test dataset with intentional label errors"""
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="test_dataset_")
    print(f"Created test dataset: {test_dir}")

    # Create subdirectories
    images_dir = os.path.join(test_dir, "images")
    labels_dir = os.path.join(test_dir, "labels")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    # Create data.yaml with 2 classes
    data_yaml = {
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': 2,
        'names': ['class1', 'class2']
    }

    with open(os.path.join(test_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    # Create some test label files with intentional errors
    test_labels = [
        # Valid labels
        "0 0.5 0.5 0.2 0.2\n",  # class 0 - valid
        "1 0.3 0.3 0.1 0.1\n",  # class 1 - valid

        # Invalid labels (what we want to catch)
        "2 0.5 0.5 0.2 0.2\n",  # class 2 - invalid (too high)
        "-1 0.3 0.3 0.1 0.1\n", # class -1 - invalid (negative)
        "1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n",  # Multiple classes in one file
    ]

    for i, label_content in enumerate(test_labels):
        with open(os.path.join(labels_dir, f"test_{i}.txt"), "w") as f:
            f.write(label_content)

    # Create a dummy image file
    with open(os.path.join(images_dir, "test_0.jpg"), "w") as f:
        f.write("dummy image")

    print("Test dataset created with intentional label errors:")
    print("- Valid class IDs: 0-1 (for 2 classes)")
    print("- Some labels use class 2 and -1 (invalid)")
    print("- Some files have multiple classes")

    return test_dir

if __name__ == "__main__":
    test_dataset = create_test_dataset()
    print(f"\nTest dataset ready at: {test_dataset}")
    print("You can now test the validation by selecting this dataset in the application.")
