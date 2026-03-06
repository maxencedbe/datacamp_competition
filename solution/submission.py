import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

class SimpleDataset(Dataset):
    def __init__(self, df, labels=None, img_dir="", size=(128, 128), transform=None):
        self.df = df
        self.labels = labels
        self.img_dir = img_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["filename"]
        path = os.path.join(self.img_dir, fname)
        try:
            img = Image.open(path).convert("RGB").resize(self.size)
            if self.transform:
                x = self.transform(img)
            else:
                x = T.ToTensor()(img)
        except Exception as e:
            x = torch.zeros((3, self.size[1], self.size[0]), dtype=torch.float32)

        if self.labels is not None:
            y = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
            return x, y
        return x

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = (128, 128)
        self.num_classes = 5
        
        try:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        except:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )
            
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4) 
        self.epochs = 10
        self.batch_size = 16

    def fit(self, X_train, y_train, train_img_dir):
        print(f"Fine-tuning sur: {self.device}")
        
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = SimpleDataset(X_train, y_train, train_img_dir, size=self.img_size, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, X_test, test_img_dir):
        print("Génération des prédictions")
        
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = SimpleDataset(X_test, None, test_img_dir, size=self.img_size, transform=test_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs)
                predictions.append(probs.cpu().numpy())

        return np.vstack(predictions)

def get_model():
    return Model()