import os
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default='./data/raw_taco/annotations.json', help='Path to annotations')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.load(f)

nr_images = len(annotations['images'])
for i in range(nr_images):
    image = annotations['images'][i]

    file_name = image['file_name']
    url_original = image['flickr_url']
    url_resized = image.get('flickr_640_url', url_original)

    file_path = os.path.join(dataset_dir, file_name)

    subdir = os.path.dirname(file_path)
    if not os.path.isdir(subdir):
        os.makedirs(subdir, exist_ok=True)

    if not os.path.isfile(file_path):
        try:
            response = requests.get(url_original, timeout=10)
            content_type = response.headers.get("Content-Type", "")

            if not content_type.startswith("image"):
                print(f"[SKIPPED] Not an image: {url_original}")
                continue

            img = Image.open(BytesIO(response.content)).convert("RGB")

            exif = img.info.get("exif")
            if exif:
                img.save(file_path, exif=exif)
            else:
                img.save(file_path)

        except Exception as e:
            print(f"[ERROR] Failed to download {url_original}: {e}")
            continue

    bar_size = 30
    x = int(bar_size * i / nr_images)
    sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i+1, nr_images))
    sys.stdout.flush()

sys.stdout.write('\nFinished\n')


from roboflow import Roboflow
rf = Roboflow(api_key="p7J2xDHPK1IXDG05wf7Y")
project = rf.workspace("masterthesis-4uovn").project("plastic-waste-qczkq")
version = project.version(2)
dataset = version.download("yolov8", location="dat")
                
               
