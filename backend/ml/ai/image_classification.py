import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageDataset(Dataset):
    """
    Custom dataset for loading images with optional labels.
    """
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            if self.labels is not None:
                return image, self.labels[idx]
            return image
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None


class ImageClassifier(pl.LightningModule):
    """
    Image classification model using a pre-trained ResNet18 backbone.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze feature extraction layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

    def predict_image(self, image_path):
        """
        Predict if an image is safe or unsafe.
        Args:
            image_path: Path to the image file.
        Returns:
            is_safe: True if the image is safe, False otherwise.
            confidence: Confidence score for the prediction.
        """
        try:
            self.eval()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = self(image_tensor)
                probabilities = output[0]
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[prediction].item()

            return prediction == 0, confidence  # 0 = safe, 1 = unsafe
        except Exception as e:
            logging.error(f"Error predicting image {image_path}: {e}")
            return None, None


def load_dataset(data_dir):
    """
    Load images and labels from a directory.
    Args:
        data_dir: Directory containing 'safe' and 'unsafe' subdirectories.
    Returns:
        ImageDataset: Dataset containing image paths and labels.
    """
    image_paths = []
    labels = []

    # Load safe images (label 0)
    safe_dir = os.path.join(data_dir, 'safe')
    if os.path.exists(safe_dir):
        for img in os.listdir(safe_dir):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(safe_dir, img))
                labels.append(0)

    # Load unsafe images (label 1)
    unsafe_dir = os.path.join(data_dir, 'unsafe')
    if os.path.exists(unsafe_dir):
        for img in os.listdir(unsafe_dir):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(unsafe_dir, img))
                labels.append(1)

    # Load adult images (label 1)
    adult_dir = os.path.join(data_dir, 'adult')
    if os.path.exists(adult_dir):
        for img in os.listdir(adult_dir):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(adult_dir, img))
                labels.append(1)

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    return ImageDataset(image_paths, torch.tensor(labels))


def train_model(train_data_dir, val_data_dir=None):
    """
    Train the image classifier.
    Args:
        train_data_dir: Directory with 'safe' and 'unsafe' subdirectories containing training images.
        val_data_dir: Optional directory with validation data structured like train_data_dir.
    Returns:
        model: Trained ImageClassifier model.
    """
    try:
        # Load datasets
        train_dataset = load_dataset(train_data_dir)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

        val_loader = None
        if val_data_dir:
            val_dataset = load_dataset(val_data_dir)
            val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

        # Initialize model
        model = ImageClassifier()

        # Setup callbacks
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor='val_loss' if val_loader else 'train_loss',
                patience=3,
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath='ml/ai/models',
                filename='image_classifier',
                monitor='val_loss' if val_loader else 'train_loss',
                mode='min'
            )
        ]

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=20,
            callbacks=callbacks,
            accelerator='auto',
            devices=1,
            logger=True
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Save final model
        model_save_path = 'ml/ai/models/image_classifier_final.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)

        logging.info(f"Model saved to {model_save_path}")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None


if __name__ == "__main__":
    # Example usage:
    train_data_dir = 'path/to/train/data'
    val_data_dir = 'path/to/val/data'
    model = train_model(train_data_dir, val_data_dir)

    if model:
        test_image_path = 'path/to/test/image.jpg'
        is_safe, confidence = model.predict_image(test_image_path)
        if is_safe is not None:
            print(f"Image is {'safe' if is_safe else 'unsafe'} with {confidence:.2%} confidence")