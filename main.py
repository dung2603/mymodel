# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.functional as F

from model.model import CombinedModel  
from dataloader import DataLoadPreprocess  # Assuming this is your dataset class
from evaluation import compute_metrics  

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader=None, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.depth_criterion = nn.L1Loss()  # Loss for depth estimation
        self.seg_criterion = nn.CrossEntropyLoss()  # Loss for segmentation
        self.num_epochs = 10  # Set the number of epochs
        self.min_depth = 0.1
        self.max_depth = 10.0
        self.scaler = torch.cuda.amp.GradScaler()

    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

    def init_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train_on_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images = batch['image'].to(self.device)
        depths = batch['depth'].to(self.device)
        
        # Đảm bảo depths có kích thước [batch_size, 1, H, W]
        if depths.ndim == 3:
            depths = depths.unsqueeze(1)

        with torch.cuda.amp.autocast():
            outputs, seg_output = self.model(images)
            # outputs: [batch_size, 1, H, W]
            # depths: [batch_size, 1, H, W]

            # Kiểm tra và điều chỉnh kích thước của outputs nếu cần
            if outputs.shape[2:] != depths.shape[2:]:
                outputs = F.interpolate(outputs, size=depths.shape[2:], mode='bilinear', align_corners=True)

            depth_loss = self.depth_criterion(outputs, depths)

            if 'segmentation' in batch:
                segmentations = batch['segmentation'].to(self.device)
                seg_loss = self.seg_criterion(seg_output, segmentations)
                total_loss = depth_loss + seg_loss
            else:
                total_loss = depth_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Return losses for logging
        loss_dict = {'depth_loss': depth_loss.item()}
        if 'segmentation' in batch:
            loss_dict['seg_loss'] = seg_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return loss_dict

    def validate_on_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)

            # Đảm bảo depths có kích thước [batch_size, 1, H, W]
            if depths.ndim == 3:
                depths = depths.unsqueeze(1)

            # Check if segmentation labels are available
            if 'segmentation' in batch:
                segmentations = batch['segmentation'].to(self.device)
            else:
                segmentations = None

            outputs, seg_output = self.model(images)
            # Kiểm tra và điều chỉnh kích thước của outputs nếu cần
            if outputs.shape[2:] != depths.shape[2:]:
                outputs = F.interpolate(outputs, size=depths.shape[2:], mode='bilinear', align_corners=True)

            depth_loss = self.depth_criterion(outputs, depths)

            # Compute segmentation loss if segmentation labels are available
            if segmentations is not None:
                seg_loss = self.seg_criterion(seg_output, segmentations)
                total_loss = depth_loss + seg_loss
            else:
                total_loss = depth_loss

            # Compute evaluation metrics
            metrics = compute_metrics(depths, outputs,
                                      min_depth=self.min_depth,
                                      max_depth=self.max_depth)

            # Return losses and metrics
            loss_dict = {'depth_loss': depth_loss.item()}
            if segmentations is not None:
                loss_dict['seg_loss'] = seg_loss.item()
            loss_dict['total_loss'] = total_loss.item()

            return {**loss_dict, **metrics}

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            train_losses = []
            self.model.train()
            for batch in tqdm(self.train_loader):
                losses = self.train_on_batch(batch)
                train_losses.append(losses['total_loss'])

            avg_train_loss = np.mean(train_losses)
            print(f"Average Training Loss: {avg_train_loss}")

            if self.test_loader is not None:
                val_losses = []
                val_metrics = []
                self.model.eval()
                for batch in self.test_loader:
                    metrics = self.validate_on_batch(batch)
                    val_losses.append(metrics['total_loss'])
                    val_metrics.append({k: metrics[k] for k in metrics if k not in ['total_loss', 'depth_loss', 'seg_loss']})

                avg_val_loss = np.mean(val_losses)
                avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
                print(f"Validation Loss: {avg_val_loss}")
                print("Validation Metrics:")
                for k, v in avg_metrics.items():
                    print(f"  {k}: {v}")

            self.scheduler.step()

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.model.load_state_dict(torch.load(filename))

def main():
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Instantiate the model
    n_classes = 40  # Adjust according to your number of segmentation classes
    model = CombinedModel(n_classes=n_classes).to(device)
    print(model)

    # Set batch size
    batch_size = 4  # Adjust batch size according to your needs and system capacity

    # Create data loaders with batch size
    train_dataset = DataLoadPreprocess(mode='train')
    test_dataset = DataLoadPreprocess(mode='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust as needed
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust as needed
        pin_memory=True
    )

    # Instantiate the trainer
    trainer = BaseTrainer(model, train_loader, test_loader, device=device)

    # Start training
    trainer.train()

    # Save the model
    trainer.save_checkpoint('combined_model.pth')

if __name__ == '__main__':
    main()
