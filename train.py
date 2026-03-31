import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing

# �����Զ���ģ��
from models.model_factory import get_model
from data.dataset import get_dataloaders
from utils.args import parse_args

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    train_loss = 0.0
    
    # ѵ���׶�
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return val_loss / len(val_loader.dataset), correct / len(val_loader.dataset)

def main(args):
    # ���ݺ�ģ�ͳ�ʼ��
    train_loader, val_loader, num_classes = get_dataloaders(args)

    # ��ʼ��ģ��
    print(f"\n��ʼ��{args.model_name.upper()}ģ��...")
    model = get_model(
        args.model_name, 
        num_classes, 
        use_pretrained=args.use_pretrained, 
        width_mult=args.width_mult, 
        device=args.device
    )
    
    # ��ӡģ����Ϣ
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"�ܲ�����: {total_params/1e6:.2f}M")
    print(f"��ѵ������: {trainable_params/1e6:.2f}M")

    # ѵ������
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # ѵ��ѭ��
    best_acc = 0.0
    print("\n��ʼѵ��...")
    for epoch in range(args.num_epochs):
        # ѵ���׶�
        train_loss_avg = train_one_epoch(model, train_loader, criterion, optimizer, args.device, epoch, args.num_epochs)

        # ��֤�׶�
        val_loss_avg, val_acc = validate(model, val_loader, criterion, args.device)
        
        # ����ѧϰ��
        scheduler.step(val_acc)
        
        # �������ģ��
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'best_{args.model_name}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"�������ģ�ͣ�׼ȷ�ʣ�{val_acc:.4f}")
        
        # ��ӡ����
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    print(f"\nѵ����ɣ������֤׼ȷ�ʣ�{best_acc:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    main(args)
