
import torch.nn as nn


# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Define the layers as per the architecture diagram
        self.features = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=3, padding=1),  # Assuming grayscale input images
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pool 2x2

            nn.Conv2d(100, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, kernel_size=3, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pool 2x2

            nn.Conv2d(200, 300, kernel_size=3, padding=1),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, kernel_size=3, padding=1),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, kernel_size=3, padding=1),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Max-pool 1x2

            nn.Conv2d(300, 400, kernel_size=3, padding=1),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, kernel_size=3, padding=1),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Max-pool 2x1

            nn.Conv2d(400, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.features(x)
        return x


# Row Encoder implementing a Bidirectional LSTM over each row of the feature grid
class RowEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(RowEncoder, self).__init__()
        # Assume feature_dim is the number of channels * width of the feature map
        # In your case, feature_dim should be 512 * 32
        self.blstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, features):
        # features shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = features.size()

        # Combine the channel and width dimensions to treat each row of pixels as a sequence
        features = features.view(batch_size, height, -1)

        # LSTM expects input of shape [batch_size, seq_len, input_size]
        # Here, seq_len is height (number of rows), and input_size is channels * width
        # Features are now of shape [batch_size, 32, 512 * 32]
        output, (hidden, cell) = self.blstm(features)

        # No need to reshape output as it is already in the shape [batch_size, seq_len, num_directions * hidden_size]
        # If you need to further process the output, you can do so depending on your requirements

        return output
