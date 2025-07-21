import numpy as np
import glob
import os

# Find all .txt files recursively
txt_files = [p for p in glob.glob('**/*.txt', recursive=True)]
print("Found text files:", txt_files)

# Load each file into a numpy array
data = {}
for fn in txt_files:
    try:
        arr = np.loadtxt(fn)
    except Exception as e:
        print(f"Skipping {fn}: cannot load ({e})")
        continue
    data[fn] = arr

# Compute pairwise RMSE between grids of the same shape
print("\nPairwise RMSE:")
files = list(data.keys())
for i in range(len(files)):
    for j in range(i+1, len(files)):
        f1, f2 = files[i], files[j]
        a, b = data[f1], data[f2]
        if a.shape != b.shape:
            print(f" - Skipping {os.path.basename(f1)} vs {os.path.basename(f2)}: shape mismatch {a.shape} vs {b.shape}")
            continue
        rmse = np.sqrt(np.mean((a - b) ** 2))
        print(f" - RMSE({os.path.basename(f1)}, {os.path.basename(f2)}) = {rmse:.6e}")
