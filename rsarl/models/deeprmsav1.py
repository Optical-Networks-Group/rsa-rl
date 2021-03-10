
import pfrl
import torch.nn as nn

class DeepRMSAv1Net(nn.Module):
    """DeepRMSAv1 Model

        paper: Deep-RMSA: A Deep-Reinforcement-Learning Routing, 
            Modulation and Spectrum Assignment Agent for Elastic Optical Networks
            (https://doi.org/10.1364/OFC.2018.W4F.2)
    """

    def __init__(self, n_slots: int, ich: int, K: int, n_edges: int):
        super().__init__()
        self.n_slots = n_slots

        # CONV
        self.conv = nn.Sequential(*[
            nn.Conv2d(ich, 1, kernel_size=(1,1), stride=(1, 1)),
            nn.ReLU(),
            # 2 conv layers with16 filters
            nn.Conv2d(1, 16, kernel_size=(n_edges,1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1,1), stride=(1, 1)),
            nn.ReLU(),
            # 2 depthwise conv layers with 1 filter
            nn.ZeroPad2d((1, 0, 0, 0)), # left, right, top, bottom
            nn.Conv2d(16, 16, kernel_size=(1,2), stride=(1, 1), groups=16),
            nn.ReLU(),
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.Conv2d(16, 16, kernel_size=(1,2), stride=(1, 1), groups=16),
            nn.ReLU(),
        ])
        # FC
        self.fc = nn.Sequential(*[
            nn.Linear(n_slots*16, 128),
            nn.ReLU(),
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, K),
        ])      

    def forward(self, x):
        h = x
        h = self.conv(h)
        h = h.view(-1, self.n_slots*16)
        h = self.fc(h)
        return pfrl.action_value.DiscreteActionValue(h)

