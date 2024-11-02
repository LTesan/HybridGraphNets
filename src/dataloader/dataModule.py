from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from src.dataloader.dataset import GraphDataset

class GraphDataModule(LightningDataModule):
    def __init__(self, args, batch_size=8, val_split=0.2, num_workers=4):
        super().__init__()
        self.args = args
        self.data_dir_train = 'data\\train\database_liver'
        self.data_dir_test = 'data\\test\extra'
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Apply transforms when initializing the dataset
        self.train_dataset = GraphDataset(self.args, self.data_dir_train)
        self.train_dataset.setup()
        
        # Compute and store the statistics in a dictionary
        stats = {
            'stats_z': self.train_dataset.compute_statistics('y'),
            'stats_f': self.train_dataset.compute_statistics('f'),
            'stats_u_w': self.train_dataset.compute_statistics('u_w', absolute=True),
            'stats_u_m': self.train_dataset.compute_statistics('u_m', absolute=True),
        }

        # Testing and validation on a different dataset with the same transforms
        self.val_dataset = GraphDataset(self.args, self.data_dir_test)
        self.val_dataset.setup()

        # Calculate lengths for validation and test splits
        val_length = int(len(self.val_dataset) * self.val_split)
        test_length = len(self.val_dataset) - val_length

        # Split the dataset into validation and test sets
        self.val_dataset, _ = random_split(self.val_dataset, [val_length, test_length])

        self.test_dataset = GraphDataset(self.args, self.data_dir_test, rollout=True)
        self.test_dataset.setup()
        
        return stats

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )
