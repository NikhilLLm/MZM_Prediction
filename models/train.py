import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=7, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_model(model, train_loader, test_loader, y_train, num_epochs=75, device='cuda'):
    try:
        model = model.to(device)
        criterion = FocalLoss(alpha=1, gamma=7)
        optimizer = optim.Adam([
            {'params': model.resnet.conv1.parameters(), 'lr': 1e-4},
            {'params': model.resnet.bn1.parameters(), 'lr': 1e-4},
            {'params': model.resnet.layer1.parameters(), 'lr': 1e-4},
            {'params': model.resnet.layer2.parameters(), 'lr': 1e-4},
            {'params': model.resnet.layer3.parameters(), 'lr': 1e-4},
            {'params': model.resnet.layer4.parameters(), 'lr': 1e-4},
            {'params': model.resnet.fc.parameters(), 'lr': 5e-4},
        ], weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_loss = val_loss / len(test_loader.dataset)
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

            # Log current learning rates before stepping the scheduler
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Learning Rates: {current_lrs}')

            scheduler.step(val_loss)

            # Check if learning rates changed after scheduler step
            new_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            if new_lrs != current_lrs:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Learning Rates updated to: {new_lrs}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        model.load_state_dict(best_model_state)
        logging.info(f"Best Validation Loss: {best_val_loss:.4f}")

        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        return accuracy, precision, recall, f1, y_pred, np.array(y_true)

    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        return None