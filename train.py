import torch
import torch.nn as nn
import os
from tqdm import tqdm
from preprocess import FaultDataset
from torch.utils.data import DataLoader
from dsdm_model import DSDM, dsdm_loss, TRAIN_CONFIG

def train_dsdm(processed_data_path, save_path="./dsdm_model"):
    os.makedirs(save_path, exist_ok=True)
    device = TRAIN_CONFIG["device"]

    print(f"使用设备: {device}")

    # 加载数据集
    print("正在加载数据集...")
    train_dataset = FaultDataset(processed_data_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, drop_last=True)
    val_dataset = FaultDataset(processed_data_path, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

    # 初始化模型
    print("正在初始化模型...")
    model = DSDM(
        num_classes=TRAIN_CONFIG["num_classes"],
        timesteps=TRAIN_CONFIG["timesteps"]
    ).to(device)

    # 优化器与学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])

    # 训练循环
    best_loss = float("inf")
    print("="*50)
    print("开始训练...")
    print("="*50)

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.shape[0]

            # 随机采样时间步
            t = torch.randint(0, TRAIN_CONFIG["timesteps"], (batch_size,), device=device).long()

            # 前向传播
            pred_noise, real_noise, enhanced_feat = model(batch_x, batch_y, t)
            loss, rec_loss, cls_loss = dsdm_loss(pred_noise, real_noise, enhanced_feat, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"总损失": f"{loss.item():.4f}", "重构损失": f"{rec_loss.item():.4f}"})

        # 验证
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                t = torch.randint(0, TRAIN_CONFIG["timesteps"], (batch_x.shape[0],), device=device).long()
                pred_noise, real_noise, enhanced_feat = model(batch_x, batch_y, t)
                loss, _, _ = dsdm_loss(pred_noise, real_noise, enhanced_feat, batch_y)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        lr_scheduler.step()

        print(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "dsdm_best.pth"))
            print(f"  -> 最优模型已保存 (Val Loss: {best_loss:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_path, "dsdm_final.pth"))
    print("="*50)
    print("训练完成！")
    print(f"模型权重保存在: {save_path}")
    print("="*50)
    return model, save_path

if __name__ == "__main__":
    processed_data_path = "./processed_data"
    if not os.path.exists(os.path.join(processed_data_path, "few_shot_X_train.npy")):
        print("错误：未找到预处理数据！请先运行 preprocess.py")
    else:
        model, model_save_path = train_dsdm(processed_data_path)
