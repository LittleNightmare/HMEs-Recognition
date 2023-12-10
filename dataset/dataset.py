import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import encode_truth, START, END


class MathExpressionDataset(Dataset):
    def __init__(self, file_path, root_dir, token_to_id, transform=None):
        self.data = []
        self.token_to_id = token_to_id
        self.transform = transform

        # Read the file and construct the data attribute
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                image_path, truth = row
                truth = encode_truth(truth, self.token_to_id)
                self.data.append((os.path.join(root_dir, image_path + '.png'), truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, truth = self.data[idx]
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        encoded_truth = [self.token_to_id[START]] + truth + [self.token_to_id[END]]
        truth_length = len(encoded_truth)
        return {
            'image': image,  # The image tensor
            'truth': {
                'text': truth,  # The LaTeX string
                'encoded': torch.tensor(encoded_truth, dtype=torch.long),  # The encoded LaTeX sequence tensor
                'length': truth_length  # The actual length of the sequence
            }
        }
