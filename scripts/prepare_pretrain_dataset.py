import os
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# Create directories inside playground/ to store pretraining data if they don't exist
PRETRAIN_DATA_DIR = '../playground/data/pretraining'

if not os.path.exists(PRETRAIN_DATA_DIR):
    os.mkdir(PRETRAIN_DATA_DIR)
    # Create subdirectories for chat and images
    os.mkdir(os.path.join(PRETRAIN_DATA_DIR, 'chat'))
    os.mkdir(os.path.join(PRETRAIN_DATA_DIR, 'images'))

# Download datasets
import requests
import zipfile
import io

# Download chat data
def download_chat():
    if os.path.exists(os.path.join(PRETRAIN_DATA_DIR, 'chat', 'blip_laion_cc_sbu_558k.json')):
        print('Chat data already exists, skipping download.')
        return

    chat_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json'
    chat_file = requests.get(chat_url)
    chat_file.raise_for_status()
    with open(os.path.join(PRETRAIN_DATA_DIR, 'chat', 'blip_laion_cc_sbu_558k.json'), 'wb') as f:
        f.write(chat_file.content)

# Download image data
def download_images():
    image_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'
    image_file = requests.get(image_url)
    image_file.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(image_file.content))
    z.extractall(os.path.join(PRETRAIN_DATA_DIR, 'images'))

    # Remove the zip file
    os.remove(os.path.join(PRETRAIN_DATA_DIR, 'images', 'images.zip'))

# Download chat and image data
download_chat()
download_images()

