import torch
import torch.nn as nn


class A2FNet(nn.Module):
    def __init__(self, num_blendshapes=61):
        super(A2FNet, self).__init__()
        self.num_blendshapes=num_blendshapes
        self.formant=nn.Sequential(
            # nn.BatchNorm2d(123),
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.articulation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4,1), stride=(4,1)),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(256, 150),
            # nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes),
            nn.ReLU()
        )
    
    def forward(self, x):
        x=self.formant(x)
        x=self.articulation(x)
        x=torch.squeeze(x)
        x=self.output(x)

        return x
