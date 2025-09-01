#!/usr/bin/env python3
"""
Fix YOLO Label Class IDs
Converts class ID 1 to class ID 0 in all label files
"""

import os
import glob

def fix_label_files(labels_dir):
    """Fix class IDs in all label files"""
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    fixed_count = 0
    total_files = len(label_files)

    print(f"Found {total_files} label files to process...")

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # Valid YOLO format line
                    class_id = int(parts[0])
                    if class_id == 1:
                        # Change class ID from 1 to 0
                        parts[0] = '0'
                        modified = True
                        print(f"Fixed {os.path.basename(label_file)}: class {class_id} -> 0")

                new_lines.append(' '.join(parts))

            if modified:
                # Write back the fixed file
                with open(label_file, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')
                fixed_count += 1

        except Exception as e:
            print(f"Error processing {label_file}: {str(e)}")

    print(f"\n✅ Fixed {fixed_count} out of {total_files} label files")
    return fixed_count

if __name__ == "__main__":
    # Path to the labels directory
    labels_dir = r"F:\Programming\Yolo_Trainer\datasets\test\labels"

    if os.path.exists(labels_dir):
        fixed_count = fix_label_files(labels_dir)
        if fixed_count > 0:
            print(f"\n🎉 Success! {fixed_count} label files have been fixed.")
            print("You can now run your training again.")
        else:
            print("\nℹ️ No files needed fixing (already correct).")
    else:
        print(f"❌ Labels directory not found: {labels_dir}")
