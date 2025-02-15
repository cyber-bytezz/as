import ezc3d
import pandas as pd
import numpy as np

# Load the C3D file
c3d_file = "input/1.c3d"  # Ensure the file is in the correct path
c3d = ezc3d.c3d(c3d_file)

# Extract point (marker) data
points = c3d["data"]["points"]  # Shape: (4, num_markers, num_frames)

# Get marker names
marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
num_frames = points.shape[2]

# Create a dictionary to store marker positions (X, Y, Z)
data_dict = {"Frame": np.arange(1, num_frames + 1)}

# Extract X, Y, and Z coordinates for each marker
for i, label in enumerate(marker_labels):
    data_dict[f"{label}_X"] = points[0, i, :]  # X-coordinates
    data_dict[f"{label}_Y"] = points[1, i, :]  # Y-coordinates
    data_dict[f"{label}_Z"] = points[2, i, :]  # Z-coordinates (Fix: Add Z extraction)

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame(data_dict)

# Save as an Excel file
output_path = "input/c3d_data.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Data extracted and saved to {output_path}")
