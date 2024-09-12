import torch
import torch.nn as nn
import torch.optim as optim
from model import ProductInfoExtractor
from dataset import get_dataloader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assuming 8 classes for now, adjust as needed
    model = ProductInfoExtractor(num_classes=8).to(device)
    
    train_loader = get_dataloader('../dataset/train.csv', '../images/train')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main()