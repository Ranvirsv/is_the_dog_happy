import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
import os, logging

logger = logging.getLogger(__name__)

## REF: https://stackoverflow.com/questions/53530751/how-make-customised-dataset-in-pytorch-for-images-and-their-masks?utm_source=chatgpt.com

class FaceBBoxEmotionDataset(Dataset):
    def __init__(self, pairs, label_map, img_size=224):
        self.pairs = pairs
        self.label_map = label_map
        self.img_size = img_size
        self.classes = list(label_map.keys())
        
        self.tf_img  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
        self.tf_mask = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, anno_path, label_name = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")

        with open(anno_path, "r") as f:
            nums = list(map(float, f.read().split()))

        if len(nums) % 4 == 0 and len(nums) >= 4:
            boxes = [nums[i:i+4] for i in range(0, len(nums), 4)]

        elif len(nums) >= 4:
            logger.warning(
                f"[BAD FORMAT] {anno_path} – "
                f"{len(nums)} values not a multiple of 4, using last 4"
            )
            boxes = [nums[-4:]]

        else:
            logger.warning(
                f"[TOO FEW POINTS] {anno_path} – "
                f"found {len(nums)} values, need at least 4, skipping"
            )
            
            W, H = img.size
            mask_arr = np.zeros((H, W), dtype=np.uint8)
            mask = Image.fromarray(mask_arr * 255)
            img_t  = self.tf_img(img)
            mask_t = self.tf_mask(mask)
            x = torch.cat([img_t, mask_t], dim=0)
            y = self.label_map[label_name]
            return x, y

        W, H = img.size
        mask_arr = np.zeros((H, W), dtype=np.uint8)

        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
            x1 = max(0, min(x1, W-1))
            x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1))
            y2 = max(0, min(y2, H-1))
            mask_arr[y1:y2, x1:x2] = 1

        mask = Image.fromarray(mask_arr * 255)
        img_t  = self.tf_img(img)
        mask_t = self.tf_mask(mask)
        x = torch.cat([img_t, mask_t], dim=0)

        y = self.label_map[label_name]
        return x, y


def make_image_anno_pairs(root_img, root_anno, subsets=("train","val","test")):
    pairs = []
    
    for sub in subsets:
        img_subdir  = os.path.join(root_img, sub)
        anno_subdir = os.path.join(root_anno, sub)
        
        for label_name in sorted(os.listdir(img_subdir)):
            img_dir  = os.path.join(img_subdir,  label_name)
            ann_dir  = os.path.join(anno_subdir, label_name)
            
            for fname in sorted(os.listdir(img_dir)):
                img_path  = os.path.join(img_dir,  fname)
                base, _   = os.path.splitext(fname)
                anno_path = os.path.join(ann_dir, base + ".txt")
                
                if not os.path.isfile(anno_path):
                    logger.warning(f"[MISSING] {anno_path} – annotation file not found, skipping")
                    continue

                with open(anno_path) as f:
                    nums = list(map(float, f.read().split()))

                if len(nums) < 4:
                    logger.warning(
                        f"[TOO FEW POINTS] {anno_path} – found {len(nums)} values, need 4, skipping"
                    )
                    continue

                pairs.append((img_path, anno_path, label_name))

                    
    return pairs

def get_loaders(data_dir, annotation_dir=None):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if annotation_dir:
        label_map   = {"angry":0,"happy":1,"relaxed":2,"sad":3}
        train_pairs = make_image_anno_pairs(data_dir, annotation_dir, ("train",))
        val_pairs   = make_image_anno_pairs(data_dir, annotation_dir, ("val",))
        test_pairs  = make_image_anno_pairs(data_dir, annotation_dir, ("test",))

        train_dataset = FaceBBoxEmotionDataset(train_pairs, label_map)
        vali_dataset = FaceBBoxEmotionDataset(val_pairs,   label_map)
        test_dataset = FaceBBoxEmotionDataset(test_pairs,  label_map)
    
    else:
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
        vali_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=test_transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=test_transform)

    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, vali_loader, test_loader, len(train_dataset.classes)