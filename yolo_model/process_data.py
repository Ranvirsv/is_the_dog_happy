import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# ========== CONFIGURATION ==========
DATA_DIR = "./Imagenet"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "Annotations")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
YOLO_DATASET_DIR = "./yolo_model/data/yolo_dataset"

# ========== CREATE YOLO DIRECTORY STRUCTURE ==========
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, "labels", split), exist_ok=True)

def is_valid_yolo_label_file(txt_path, num_classes=1):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Invalid format in: {txt_path} → {line.strip()}")
            return False
        try:
            class_id = int(parts[0])
            if not (0 <= class_id < num_classes):
                print(f"Invalid class ID in: {txt_path}")
                return False

            bbox = [float(x) for x in parts[1:]]
            if any([x <= 0.0 or x > 1.0 for x in bbox]):
                print(f"Invalid normalized bbox in: {txt_path} → {bbox}")
                return False
        except ValueError:
            print(f"Non-numeric value in: {txt_path} → {line.strip()}")
            return False

    return True

# ========== UTILITY FUNCTION ==========
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

# ========== STEP 1: GATHER ALL IMAGE FILES ==========
image_files = []
for dirpath, _, filenames in os.walk(IMAGES_DIR):
    for file in filenames:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(dirpath, file))

print(f"Total images found: {len(image_files)}")

# ========== STEP 2: SPLIT INTO TRAIN AND VAL ==========
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# ========== STEP 3: GET XML PATH FROM IMAGE ==========
def get_annotation_path(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    folder_name = os.path.basename(os.path.dirname(image_path))
    annotation_path = os.path.join(ANNOTATIONS_DIR, folder_name, image_name)
    
    if os.path.exists(annotation_path):
        return annotation_path
    elif os.path.exists(annotation_path + ".xml"):
        return annotation_path + ".xml"
    else:
        return None

# ========== STEP 4: PARSE & CONVERT FUNCTION ==========
def process_images(image_list, split):
    for image_path in image_list:
        annotation_path = get_annotation_path(image_path)
        if annotation_path is None:
            print(f"No annotation found for {image_path}")
            continue

        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Failed to parse XML {annotation_path}: {e}")
            continue

        size_tag = root.find("size")
        w = int(size_tag.find("width").text)
        h = int(size_tag.find("height").text)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(YOLO_DATASET_DIR, "labels", split, base_name + ".txt")
        with open(txt_path, "w") as f:
            for obj in root.findall("object"):
                cls_id = 0

                xml_box = obj.find("bndbox")
                box = (
                    int(xml_box.find("xmin").text),
                    int(xml_box.find("ymin").text),
                    int(xml_box.find("xmax").text),
                    int(xml_box.find("ymax").text)
                )
                yolo_box = convert_bbox((w, h), box)
                f.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in yolo_box)}\n")

        # Immediately validate the file
        if not is_valid_yolo_label_file(txt_path, num_classes=1):
            # Remove label and image if invalid
            os.remove(txt_path)
            os.remove(image_path)
            continue

        out_image_path = os.path.join(YOLO_DATASET_DIR, "images", split, os.path.basename(image_path))
        shutil.copy(image_path, out_image_path)

# ========== STEP 5: PROCESS ==========
process_images(train_images, "train")
process_images(val_images, "val")

print("YOLOv8 dataset prepared successfully!")