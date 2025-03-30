from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_loaders(data_dir):
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

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
    vali_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=test_transform)
    test_datset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=test_transform)

    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_datset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, vali_loader, test_loader, len(train_dataset.classes)