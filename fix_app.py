#!/usr/bin/env python3

# Script to fix the broken app.py by removing the old coordinate system

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the lines to remove (from after the "st.warning('⚠️ No boxes to sync to canvas!')" 
# to before "with tab2:")

# Find where the good code ends
good_end = -1
for i, line in enumerate(lines):
    if "st.warning('⚠️ No boxes to sync to canvas!')" in line:
        good_end = i + 1  # Include the blank line after
        break

# Find where the good code starts again  
good_start = -1
for i, line in enumerate(lines):
    if line.strip() == "with tab2:":
        good_start = i
        break

print(f"Found good_end at line {good_end + 1}")
print(f"Found good_start at line {good_start + 1}")

if good_end != -1 and good_start != -1:
    # Create new file content
    new_lines = lines[:good_end]
    new_lines.append("\n")
    new_lines.append("        else:\n")  
    new_lines.append("            st.info(\"📷 Please select an image from the dataset to start annotating!\")\n")
    new_lines.append("\n")
    new_lines.extend(lines[good_start:])
    
    # Write the fixed file
    with open('app_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("Fixed file created as app_fixed.py")
    print(f"Removed lines {good_end + 1} to {good_start}")
else:
    print("Could not find the boundaries to fix")
