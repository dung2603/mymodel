# dataloader.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def remove_leading_slash(s):
    return s.lstrip('/\\')

class DataLoadPreprocess(Dataset):
    def __init__(self, mode, **kwargs):
        self.mode = mode

        # Define image transform
        self.image_transform = transforms.Compose([
            transforms.Pad((0, 0, 14 - (640 % 14), 14 - (480 % 14))),  # Add padding to make dimensions divisible by 14
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Define depth transform
        self.depth_transform = transforms.Compose([
            transforms.Pad((0, 0, 14 - (640 % 14), 14 - (480 % 14))),
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Paths to data (Update these paths according to your dataset location)
        base_path = r'C:\filemohinh\modelmoi\train'  # Base directory for your data

        if mode == 'train':
            self.data_path = base_path  # Path to training images and depths
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_train_files_with_gt.txt')  # Path to your train.txt file
        else:
            self.data_path = base_path  # Path to test images and depths
            self.filenames_file = os.path.join(base_path, 'nyudepthv2_test_files_with_gt.txt')  # Path to your test.txt file

        # Read the list of filenames
        with open(self.filenames_file, 'r') as f:
            self.filenames = f.readlines()

        # Depth limits
        self.min_depth = 0.1
        self.max_depth = 10.0

    def __getitem__(self, idx):
        sample_line = self.filenames[idx].strip()
        sample_items = sample_line.split()

        if len(sample_items) < 3:
            raise ValueError(f"Invalid line in filename file: {sample_line}")

        image_rel_path = sample_items[0]
        depth_rel_path = sample_items[1]
        focal = torch.tensor(float(sample_items[2]), dtype=torch.float32)

        # Construct full paths
        image_path = os.path.join(self.data_path, remove_leading_slash(image_rel_path))
        depth_path = os.path.join(self.data_path, remove_leading_slash(depth_rel_path))

        # Read image and depth
        image = Image.open(image_path).convert('RGB')
        depth_gt = Image.open(depth_path)

        # Apply transforms
        image = self.image_transform(image)
        depth_gt = self.depth_transform(depth_gt).squeeze(0) / 1000.0  # Convert from mm to meters

        # Create mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        mask = mask.float().unsqueeze(0)  # Add channel dimension

        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.filenames)

# If you want to test the dataset
if __name__ == '__main__':
    dataset = DataLoadPreprocess(mode='train')

    # Get one sample
    sample = dataset[0]

    print("Image shape:", sample['image'].shape)
    print("Depth shape:", sample['depth'].shape)
    print("Focal length:", sample['focal'])
    print("Mask shape:", sample['mask'].shape)
