import os
import yaml

def fix_data_yaml(dataset_path):
    """Fix the data.yaml file to use relative paths"""
    yaml_path = os.path.join(dataset_path, "data.yaml")

    if os.path.exists(yaml_path):
        # Read current content
        with open(yaml_path, 'r') as f:
            content = f.read()

        # Check if it has the old format
        if 'datasets/' in content:
            print(f"Fixing {yaml_path}")

            # Create new content with relative paths
            new_content = """train: images
val: images
test: images

nc: 1
names: ['object']
"""

            # Write the fixed content
            with open(yaml_path, 'w') as f:
                f.write(new_content)

            print(f"Fixed {yaml_path}")
        else:
            print(f"{yaml_path} already has correct format")

# Fix all datasets
datasets_dir = "datasets"
if os.path.exists(datasets_dir):
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if os.path.isdir(dataset_path):
            fix_data_yaml(dataset_path)

print("All data.yaml files have been fixed!")
