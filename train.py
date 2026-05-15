import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import argparse
from tqdm import tqdm

from data.dataset import get_dataloader
from models.model_factory import get_mobilenet_v2, get_mobilenet_v3_small, get_mobilenet_v3_large

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, save_dir='checkpoints'):
    since = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 使用 tqdm 添加迭代进度条
            pbar = tqdm(dataloaders[phase], desc=f'Epoch {epoch}/{num_epochs - 1} [{phase}]', unit='batch', leave=False)
            
            # 迭代数据
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只有在训练阶段才跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才进行后向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条后缀显示 loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度拷贝最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    return model

def main():
    parser = argparse.ArgumentParser(description='Train MobileNet with PyTorch')
    parser.add_argument('--data_dir', type=str, default='D:\\github\\Datasets\\step3_6.分类-数据集划分', help='dataset path')
    parser.add_argument('--model', type=str, default='v2', choices=['v2', 'v3_small', 'v3_large'], help='model version')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')
    parser.add_argument('--save_dir', type=str, default='train_output', help='save directory')
    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据
    dataloaders, dataset_sizes, class_names = get_dataloader(args.data_dir, args.batch_size)

    # 获取模型
    if args.model == 'v2':
        model_ft = get_mobilenet_v2(num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.model == 'v3_small':
        model_ft = get_mobilenet_v3_small(num_classes=args.num_classes, pretrained=args.pretrained)
    else:
        model_ft = get_mobilenet_v3_large(num_classes=args.num_classes, pretrained=args.pretrained)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # 观察所有参数都被优化
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)

    # 每 7 个 epochs 衰减一次权重 0.1 倍
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 开始训练s
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, device, num_epochs=args.epochs, save_dir=args.save_dir)

if __name__ == '__main__':
    # Windows 平台上使用多线程 DataLoader 需要在 main 中调用，并建议设置支持
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
    main()
