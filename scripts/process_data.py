import json
import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path('../data')
RAW_DIR = BASE_DIR / 'raw_taco'
DEST_DIR = BASE_DIR / 'processed_yolo'
ANNOTATIONS_FILE = RAW_DIR / 'annotations.json'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] normalisé (0-1)
    """
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return x_center, y_center, width, height

def main():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    
    for split in ['train', 'val', 'test']:
        (DEST_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    img_to_anns = {img['id']: [] for img in images}
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    random.seed(42)
    random.shuffle(images)

    num_imgs = len(images)
    num_train = int(num_imgs * TRAIN_RATIO)
    num_val = int(num_imgs * VAL_RATIO)
    
    train_imgs = images[:num_train]
    val_imgs = images[num_train:num_train+num_val]
    test_imgs = images[num_train+num_val:]

    splits = [
        ('train', train_imgs),
        ('val', val_imgs),
        ('test', test_imgs)
    ]

    print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    for split_name, split_images in splits:
        print(f"Create {split_name}")
        
        for img_info in tqdm(split_images):
            img_id = img_info['id']
            filename_original = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']
            
            filename_clean = filename_original.replace('/', '_')
            
            src_path = RAW_DIR / filename_original
            dst_path = DEST_DIR / split_name / 'images' / filename_clean
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
            else:
                continue

            label_filename = filename_clean.rsplit('.', 1)[0] + '.txt'
            dst_label_path = DEST_DIR / split_name / 'labels' / label_filename
            
            img_anns = img_to_anns[img_id]
            
            with open(dst_label_path, 'w') as f_label:
                for ann in img_anns:
                    class_id = 0 
                    
                    bbox = ann['bbox']
                    yolo_bbox = convert_bbox_coco_to_yolo(bbox, img_w, img_h)
                    
                    f_label.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    yaml_content = {
        'path': str(DEST_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['Dechet']
    }

    with open(DEST_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

if __name__ == "__main__":
    main()
