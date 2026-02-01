import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Swish(pl.LightningModule):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ShallowCNN(pl.LightningModule):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        c_hid1 = hidden_features
        c_hid2 = hidden_features * 2
        c_hid3 = hidden_features * 4

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hid3, num_classes)
        )

    # returns batchsize x num_classes logits
    def get_logits(self, x) -> torch.Tensor: 
        # TODO (3.2): Implement classification procedure that outputs the logits across the classes
        #  Consider using F.adaptive_avg_pool2d to convert between the 2D features and a linear representation.
        # x = F.adaptive_avg_pool2d(x, (64, 64))

        x = self.cnn_layers(x)
        if torch.isnan(x).sum() > 0:
            a = 1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc_layers(x)
        return x

    # returns energy values
    def forward(self, x, y=None) -> torch.Tensor:
        # TODO (3.2): Implement forward function for (1) Unconditional JEM (EBM), (2) Conditional JEM.
        #  (You can also reuse your implementation of 'self.get_logits(x)' if this helps you.)
   
        logits = self.get_logits(x)
        if y is None:
            # Unconditional JEM (EBM)
            e = torch.logsumexp(logits, dim=1)
            return e

        else:
            # Conditional JEM
            
            return logits[range(len(y)),y]