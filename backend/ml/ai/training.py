import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from .dataset import URLDataset, generate_dataset  # Relative import

class URLClassifier(pl.LightningModule):
    def __init__(self, input_size=8):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['features'], batch['label']
        y_hat = self(x)
        loss = torch.nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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

def train():
    df = generate_dataset()
    df.to_csv('url_dataset.csv', index=False)
    
    urls = df['url'].values
    labels = df['is_blocked'].values
    
    dataset = URLDataset(urls, labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    
    model = URLClassifier()
    
    callbacks = [
        EarlyStopping(monitor='train_loss', patience=3),
        ModelCheckpoint(
            dirpath='ml/ai/models',
            filename='url_classifier',
            monitor='train_loss'
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        accelerator='auto',
        devices=1
    )
    
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'ml/ai/models/url_classifier_final.pth')
    
    return model
