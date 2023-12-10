# Step 3: Define the Data Module
import os

from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as pl

from dataset.dataset import MathExpressionDataset
from dataset.transformations import AddRandomNoise
from utils import load_vocab, PAD, collate_batch


class MathExpressionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, tokens_file, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.tokens_file = tokens_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.token_to_id, self.id_to_token = load_vocab(os.path.join(data_dir, tokens_file))
        self.pad_index = self.token_to_id[PAD]
        # simply transform

        self.transform_base = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),  # Random rotation between -15 to 15 degrees
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random scaling between 80% to 100%
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor before applying custom noise
            AddRandomNoise(intensity=0.05),  # Add random noise
            transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        # Define paths for train and validation data files
        if stage == 'fit' or stage is None:
            self.train_dataset = MathExpressionDataset(
                file_path=os.path.join(self.data_dir, 'gt_split/train.tsv'),
                root_dir=os.path.join(self.data_dir, 'train'),
                token_to_id=self.token_to_id,
                transform=self.transform
            )

            self.val_dataset = MathExpressionDataset(
                file_path=os.path.join(self.data_dir, 'gt_split/validation.tsv'),
                root_dir=os.path.join(self.data_dir, 'train'),
                token_to_id=self.token_to_id,
                transform=self.transform_base
            )

        if stage == 'test' or stage is None:
            self.test_datasets = {
                '2013': MathExpressionDataset(
                    file_path=os.path.join(self.data_dir, 'groundtruth_2013.tsv'),
                    root_dir=os.path.join(self.data_dir, 'test/2013'),
                    token_to_id=self.token_to_id,
                    transform=self.transform_base
                ),
                '2014': MathExpressionDataset(
                    file_path=os.path.join(self.data_dir, 'groundtruth_2014.tsv'),
                    root_dir=os.path.join(self.data_dir, 'test/2014'),
                    token_to_id=self.token_to_id,
                    transform=self.transform_base
                ),
                '2016': MathExpressionDataset(
                    file_path=os.path.join(self.data_dir, 'groundtruth_2016.tsv'),
                    root_dir=os.path.join(self.data_dir, 'test/2016'),
                    token_to_id=self.token_to_id,
                    transform=self.transform_base
                ),
            }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_batch,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return {year: DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_batch, num_workers=self.num_workers)
                for year, dataset in self.test_datasets.items()}
