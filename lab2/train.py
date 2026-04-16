import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

from models.exposure_synthesizer import ExposureSynthesizer
from utils.dataset import HDREyeDataset
from utils.hdr import apply_exposure_adjustment


def create_exposure_targets(ldr_tensor: torch.Tensor, ev: float = 2.7):
    """Create exposure adjusted targets from LDR image."""
    ldr_np = (ldr_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
    
    underexposed = apply_exposure_adjustment(ldr_np, -ev)
    overexposed = apply_exposure_adjustment(ldr_np, ev)
    
    underexposed = torch.from_numpy(underexposed).float() / 255.0
    overexposed = torch.from_numpy(overexposed).float() / 255.0
    
    underexposed = underexposed.permute(2, 0, 1)
    overexposed = overexposed.permute(2, 0, 1)
    
    return torch.cat([underexposed, overexposed], dim=0)


def train_one_epoch(model, dataloader, optimizer, device, ev=2.7):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        ldr_tensor = batch[0].to(device)
        
        target = create_exposure_targets(ldr_tensor[0], ev)
        for i in range(1, ldr_tensor.size(0)):
            target = torch.cat([target, create_exposure_targets(ldr_tensor[i], ev)], dim=0)
        
        optimizer.zero_grad()
        output = model(ldr_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ExposureSynthesizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_dataset = HDREyeDataset(args.data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.ev)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
    
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HDR-Eye dataset')
    parser.add_argument('--output', type=str, default='models/exposure_model.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ev', type=float, default=2.7, help='Exposure value for synthesis')
    args = parser.parse_args()
    
    Path('checkpoints').mkdir(exist_ok=True)
    
    torch.set_num_threads(1)
    main(args)
