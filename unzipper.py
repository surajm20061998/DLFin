import zipfile
import os

# Directory where this script lives
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_FILENAME = "ys2krqw.zip"
ZIP_PATH = os.path.join(PROJECT_DIR, ZIP_FILENAME)

OUT_DIR = os.path.join(PROJECT_DIR, "images")  # extracted folder

print(f"Extracting {ZIP_PATH} into {OUT_DIR} ...")

os.makedirs(OUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(OUT_DIR)

print("Extraction complete.")
print(f"Images available at: {OUT_DIR}")