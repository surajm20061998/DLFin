import os
from huggingface_hub import hf_hub_download

# Directory where this script lives (your project folder)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# HuggingFace dataset repo and ZIP filename
REPO_ID = "sm12377/trImgs"
ZIP_FILENAME = "ys2krqw.zip"

print("Downloading ZIP into project directory:", PROJECT_DIR)

zip_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=ZIP_FILENAME,
    repo_type="dataset",
    local_dir=PROJECT_DIR,               # <<<<<< save here!
    local_dir_use_symlinks=False         # <<<<<< ensures actual file, not symlink
)

print("Download complete!")
print("ZIP saved at:", zip_path)