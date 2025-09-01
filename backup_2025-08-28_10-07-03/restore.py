#!/usr/bin/env python3
"""
YOLO Trainer - Quick Restore Script
Restores the project from backup_2025-08-28_10-07-03
"""

import os
import shutil
from pathlib import Path

def restore_backup():
    """Restore project from backup"""
    backup_dir = "backup_2025-08-28_10-07-03"
    project_root = Path(__file__).parent

    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory '{backup_dir}' not found!")
        return False

    print("🔄 Starting project restoration...")
    print(f"📂 Restoring from: {backup_dir}")

    # Files to restore
    restore_items = [
        "dataset_creator.py",
        "README_Dataset_Creator.md",
        "*.bat",
        "*.pt"
    ]

    # Directories to restore
    restore_dirs = [
        "datasets",
        "models"
    ]

    # Restore individual files
    for item in restore_items:
        if "*" in item:
            # Handle wildcards
            import glob
            matches = glob.glob(os.path.join(backup_dir, item))
            for match in matches:
                filename = os.path.basename(match)
                dest = project_root / filename
                print(f"📄 Restoring: {filename}")
                shutil.copy2(match, dest)
        else:
            src = Path(backup_dir) / item
            if src.exists():
                dest = project_root / item
                print(f"📄 Restoring: {item}")
                shutil.copy2(src, dest)

    # Restore directories
    for dir_name in restore_dirs:
        src = Path(backup_dir) / dir_name
        dest = project_root / dir_name
        if src.exists():
            if dest.exists():
                print(f"🗑️ Removing existing: {dir_name}")
                shutil.rmtree(dest)
            print(f"📁 Restoring directory: {dir_name}")
            shutil.copytree(src, dest)

    # Restore training directories
    import glob
    training_dirs = glob.glob(os.path.join(backup_dir, "training_*"))
    for training_dir in training_dirs:
        dir_name = os.path.basename(training_dir)
        dest = project_root / dir_name
        if dest.exists():
            shutil.rmtree(dest)
        print(f"📁 Restoring training data: {dir_name}")
        shutil.copytree(training_dir, dest)

    print("✅ Restoration completed successfully!")
    print("\n🚀 You can now run the application with:")
    print("   python dataset_creator.py")

    return True

if __name__ == "__main__":
    print("🛡️ YOLO Trainer - Project Restore Utility")
    print("=" * 50)

    if restore_backup():
        print("\n🎉 Project successfully restored!")
        print("📋 Your application is ready to use.")
    else:
        print("\n❌ Restoration failed. Please check the backup directory.")
