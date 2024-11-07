import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from depth_anything.dpt import DPT_DINOv2  
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Tham số huấn luyện
num_epochs = 20
learning_rate = 1e-4
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa các hàm đánh giá hiệu suất
def compute_errors(gt, pred):
    """Tính toán các chỉ số RMSE, log10, a1, a2, a3, abs_rel."""
    assert gt.shape == pred.shape, f"Kích thước không khớp: gt {gt.shape}, pred {pred.shape}"


    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    log10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred)))
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    
    return rmse.item(), log10.item(), a1.item(), a2.item(), a3.item(), abs_rel.item()

# Hàm huấn luyện một epoch với tính toán chỉ số đánh gi
scaler = GradScaler()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    rmse_list, log10_list, a1_list, a2_list, a3_list, abs_rel_list = [], [], [], [], [], []

    for batch_idx, (images, depths, focals) in enumerate(tqdm(train_loader)):
        images, depths, focals = images.to(device), depths.to(device), focals.to(device)

        # Debug: In kích thước ban đầu của images và depths
        print(f"[Batch {batch_idx}] Kích thước ban đầu của images: {images.shape}")
        print(f"[Batch {batch_idx}] Kích thước ban đầu của depths: {depths.shape}")

        # Loại bỏ chiều dư thừa của `depths` nếu tồn tại
        if depths.dim() == 5 and depths.shape[2] == 1:
            depths = depths.squeeze(2)  # Kích thước sẽ trở thành [8, 1, 480, 640]
            print(f"[Batch {batch_idx}] Kích thước của depths sau khi loại bỏ chiều dư thừa: {depths.shape}")

        # Kiểm tra lại sau khi squeeze, nếu vẫn có vấn đề về kích thước thì in ra
        if depths.dim() != 4:
            print(f"[Batch {batch_idx}] Kích thước không hợp lệ của depths: {depths.shape}. Phải có 4 chiều.")
            continue  # Bỏ qua batch nếu `depths` không hợp lệ

        # Forward pass với độ chính xác hỗn hợp
        optimizer.zero_grad()
        loss = None  # Khởi tạo `loss` để tránh UnboundLocalError
        with autocast():
            try:
                outputs, _ = model(images)  # Lấy đầu ra độ sâu và bỏ qua đặc trưng ngữ nghĩa
            except Exception as e:
                print(f"[Batch {batch_idx}] Lỗi khi lấy đầu ra từ mô hình: {e}")
                continue  # Bỏ qua batch này nếu xảy ra lỗi

            # Thêm chiều số lượng kênh cho `outputs` nếu cần
            outputs = outputs.unsqueeze(1)  # Kích thước sẽ thành [8, 1, 224, 224]
            print(f"[Batch {batch_idx}] Kích thước của outputs sau khi thêm chiều: {outputs.shape}")

            # Thay đổi kích thước `outputs` để khớp với `depths`
            try:
                outputs_resized = F.interpolate(outputs, size=(depths.shape[-2], depths.shape[-1]), mode="bilinear", align_corners=True)
                print(f"[Batch {batch_idx}] Kích thước của outputs sau khi nội suy: {outputs_resized.shape}")
            except Exception as e:
                print(f"[Batch {batch_idx}] Lỗi khi thay đổi kích thước của outputs: {e}")
                continue  # Bỏ qua batch này nếu xảy ra lỗi

            # Tính toán mất mát
            try:
                loss = criterion(outputs_resized, depths)
            except Exception as e:
                print(f"[Batch {batch_idx}] Lỗi khi tính toán mất mát: {e}")
                continue  # Bỏ qua batch này nếu xảy ra lỗi

        # Backward pass với độ chính xác hỗn hợp
        if loss is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Cộng dồn mất mát nếu `loss` đã được gán giá trị
            running_loss += loss.item()

        # Giải phóng bộ nhớ không cần thiết
        del images, focals, outputs, loss
        torch.cuda.empty_cache()

        # Tính toán chỉ số đánh giá mà không yêu cầu gradient
        if 'outputs_resized' in locals():
            with torch.no_grad():
                try:
                    rmse, log10, a1, a2, a3, abs_rel = compute_errors(depths, outputs_resized)
                except AssertionError as e:
                    print(f"[Batch {batch_idx}] Lỗi khi tính toán chỉ số đánh giá: {e}")
                    print(f"[Batch {batch_idx}] Kích thước của depths: {depths.shape}, outputs_resized: {outputs_resized.shape}")
                    continue  # Bỏ qua batch này nếu xảy ra lỗi

            # Cộng dồn chỉ số đánh giá
            rmse_list.append(rmse)
            log10_list.append(log10)
            a1_list.append(a1)
            a2_list.append(a2)
            a3_list.append(a3)
            abs_rel_list.append(abs_rel)

    # Tính giá trị trung bình cho các chỉ số
    if len(rmse_list) > 0:
        avg_rmse = np.mean(rmse_list)
        avg_log10 = np.mean(log10_list)
        avg_a1 = np.mean(a1_list)
        avg_a2 = np.mean(a2_list)
        avg_a3 = np.mean(a3_list)
        avg_abs_rel = np.mean(abs_rel_list)
    else:
        avg_rmse = avg_log10 = avg_a1 = avg_a2 = avg_a3 = avg_abs_rel = float('nan')

    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
    
    return avg_loss, avg_rmse, avg_log10, avg_a1, avg_a2, avg_a3, avg_abs_rel



def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        avg_loss, avg_rmse, avg_log10, avg_a1, avg_a2, avg_a3, avg_abs_rel = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"RMSE: {avg_rmse:.4f}, log10: {avg_log10:.4f}, a1: {avg_a1:.4f}, a2: {avg_a2:.4f}, a3: {avg_a3:.4f}, Abs Rel: {avg_abs_rel:.4f}")
        
        # Lưu mô hình sau mỗi vài epoch
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'dpt_dinov2_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")

# Main code để chạy huấn luyện
if __name__ == '__main__':
    # Tạo Dataloader
    train_loader = create_dataloader(txt_file=r"C:\filemohinh\modelmoi\train_test_inputs\nyudepthv2_train_files_with_gt.txt", root_dir=r"C:\filemohinh\modelmoi\train", batch_size=batch_size)
    test_loader = create_dataloader(txt_file=r"C:\filemohinh\modelmoi\train_test_inputs\nyudepthv2_test_files_with_gt.txt", root_dir=r"C:\filemohinh\modelmoi\train", batch_size=batch_size)
    
    # Khởi tạo mô hình
    model = DPT_DINOv2(encoder='vitl', localhub=True).to(device)
    
    # Định nghĩa hàm mất mát và trình tối ưu hóa
    criterion = nn.MSELoss()  # Mất mát bình phương sai số trung bình cho bài toán ước lượng độ sâu
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Chạy huấn luyện
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
