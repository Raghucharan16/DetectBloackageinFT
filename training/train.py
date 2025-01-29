import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from data_loader import UltrasoundDataset
from config import Hyperparameters

def train():
    cfg = Hyperparameters()
    
    # Data pipeline
    train_ds = UltrasoundDataset(cfg, mode='train')
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # Model
    model = MedicalX3D().cuda()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.max_lr, 
        weight_decay=1e-5
    )
    
    # Scheduler
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=cfg.max_lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs
    )
    
    # Loss function (Focal Loss for class imbalance)
    loss_fn = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        'FocalLoss',
        gamma=2,
        reduction='mean'
    )
    
    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            clips, labels = batch
            outputs = model(clips.cuda())
            
            loss = loss_fn(outputs, labels.cuda())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # Validation
        val_acc = validate(model, cfg)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")
