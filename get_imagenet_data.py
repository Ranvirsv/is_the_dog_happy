import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import random
from sklearn.model_selection import train_test_split

class DogFaceDataset(Dataset):
    def __init__(self, images, annotations, transform=None):
        """
        Args:
            images (list): List of image file paths.
            annotations (list): List of annotation file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.annotations = annotations
        self.transform = transform
        
    def parse_xml(self, xml_file):
        """Parse XML annotation file to extract bounding box information."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes if boxes else None  # Return None if no bounding box is found

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = self.annotations[idx]
        
        image = Image.open(img_path).convert("RGB")
        boxes = self.parse_xml(annotation_path)
        
        target = {}
        if boxes:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor([1] * len(boxes), dtype=torch.int64)  # Class 1 for dog
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def split_dataset(image_paths, annotation_paths, val_size=0.1, test_size=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        image_paths (list): List of image file paths.
        annotation_paths (list): List of annotation file paths.
        val_size (float): Proportion of data to be used for validation.
        test_size (float): Proportion of data to be used for testing.
        
    Returns:
        Tuple of lists containing image and annotation file paths for train, val, and test sets.
    """
    # Split into train + temp (val + test)
    train_imgs, temp_imgs, train_ann, temp_ann = train_test_split(image_paths, annotation_paths, test_size=val_size + test_size, random_state=42)
    
    # Split temp into val and test
    val_imgs, test_imgs, val_ann, test_ann = train_test_split(temp_imgs, temp_ann, test_size=test_size / (val_size + test_size), random_state=42)
    
    return train_imgs, train_ann, val_imgs, val_ann, test_imgs, test_ann

def get_loaders(data_dir, batch_size=16, num_workers=4):
    image_dir = os.path.join(data_dir, "Images")
    annotation_dir = os.path.join(data_dir, "Annotations")

    print(image_dir)
    
    # Get the list of image and annotation file paths
    image_paths = []
    annotation_paths = []
    
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        annotation_path = os.path.join(annotation_dir, category)
        
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                annotation_file = os.path.join(annotation_path, img_name.replace('.jpg', ''))
                
                if os.path.exists(annotation_file):
                    image_paths.append(img_path)
                    annotation_paths.append(annotation_file)
    
    # Split the dataset into train, validation, and test sets
    train_imgs, train_ann, val_imgs, val_ann, test_imgs, test_ann = split_dataset(image_paths, annotation_paths)

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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets for each split
    train_dataset = DogFaceDataset(train_imgs, train_ann, transform=train_transform)
    val_dataset = DogFaceDataset(val_imgs, val_ann, transform=test_transform)
    test_dataset = DogFaceDataset(test_imgs, test_ann, transform=val_transform)

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_loaders("/Users/devpatel/Desktop/IUB/Study/Computer Vision/Project/is_the_dog_happy/bounding_box_data")
