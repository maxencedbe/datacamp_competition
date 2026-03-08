import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")


class RetinopathyDataset(Dataset):
    def __init__(self, df, labels=None, img_dir="", size=(224, 224), transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True) if labels is not None else None
        self.img_dir = img_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["filename"]
        path = os.path.join(self.img_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(self.size, Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", self.size, (0, 0, 0))

        if self.transform:
            x = self.transform(img)
        else:
            x = T.ToTensor()(img)

        if self.labels is not None:
            y = torch.tensor(self.labels.iloc[idx].values.astype(np.float32))
            return x, y
        return x


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = (224, 224)
        self.num_classes = 5
        self.epochs = 15
        self.batch_size = 16
        self.lr = 3e-4
        self.model = None

    def _build_model(self):
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, self.num_classes),
        )
        return model.to(self.device)

    def _compute_class_weights(self, y_train):
        """Compute inverse-frequency weights for the loss to handle imbalance."""
        label_cols = ["0", "1", "2", "3", "4"]
        counts = y_train[label_cols].sum(axis=0).values.astype(np.float32)
        # Inverse frequency, normalized
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def fit(self, X_train, y_train, train_img_dir):
        print(f"Training on: {self.device}")

        self.model = self._build_model()

        # Class weights for imbalanced data
        class_weights = self._compute_class_weights(y_train)

        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        label_cols = ["0", "1", "2", "3", "4"]
        dataset = RetinopathyDataset(
            X_train, y_train[label_cols], train_img_dir,
            size=self.img_size, transform=train_transform,
        )

        # Weighted sampler to oversample minority classes
        class_indices = y_train[label_cols].values.argmax(axis=1)
        class_counts = np.bincount(class_indices, minlength=self.num_classes).astype(np.float32)
        sample_weights = 1.0 / (class_counts[class_indices] + 1e-6)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X_test, test_img_dir):
        print("Generating predictions")

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        dataset = RetinopathyDataset(
            X_test, None, test_img_dir,
            size=self.img_size, transform=test_transform,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

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
