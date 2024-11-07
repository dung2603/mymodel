import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class DepthDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Khởi tạo dataset từ file .txt.

        Args:
            txt_file (str): Đường dẫn tới file .txt chứa thông tin đường dẫn ảnh và tiêu cự.
            root_dir (str): Thư mục gốc chứa dữ liệu.
            transform (callable, optional): Các phép biến đổi ảnh cho ảnh RGB và ảnh độ sâu.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Đọc file .txt để lấy các đường dẫn ảnh và tiêu cự
        with open(txt_file, 'r') as f:
            self.data = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tách các thành phần: đường dẫn ảnh, đường dẫn độ sâu, và tiêu cự
        img_path, depth_path, focal = self.data[idx]
        
        # Ghép với thư mục gốc
        img_path = os.path.join(self.root_dir, img_path)
        depth_path = os.path.join(self.root_dir, depth_path)

        # Đọc ảnh RGB và ảnh độ sâu
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path)

        # Áp dụng các phép biến đổi, nếu có
        if self.transform:
            image = self.transform(image)

        # Chuyển đổi ảnh độ sâu thành tensor và chuyển thành kiểu float
        depth = transforms.ToTensor()(depth).float()  # Đảm bảo `depth` có kiểu float
        depth = torch.unsqueeze(depth, 0)  # Để độ sâu có dạng [1, H, W]

        # Chuyển tiêu cự thành float và đưa vào tensor
        focal = torch.tensor(float(focal))

        return image, depth, focal

# Định nghĩa các phép biến đổi cho ảnh RGB
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa ảnh RGB
])

# Hàm tạo dataloader
def create_dataloader(txt_file, root_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = DepthDataset(txt_file=txt_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
