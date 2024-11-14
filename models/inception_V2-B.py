import torch
import torch.nn as nn

class InceptionV2_B(nn.Module):
    def __init__(self, in_channels, f1, f3_r, f3, f5_r, f5, pool_proj):
        super(InceptionV2_B, self).__init__()
        
        # 1x1
        self.layer1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        
        
        # 1x1 -> 3x3(3 x 1, 1 x 3)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, f3_r, kernel_size=1),
            nn.Conv2d(f3_r, f3, kernel_size=(1, 3), padding=(0,1)),
            nn.Conv2d(f3, f3, kernel_size=(3,1), padding=(1,0))
        )
        
        
        # 1x1 -> 5x5(5 X 1, 1 X 5)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, f5_r, kernel_size=1),
            nn.Conv2d(f5_r, f5, kernel_size=(1,5), padding=(0, 2)),
            nn.Conv2d(f5, f5, kernel_size=(5,1), padding=(2,0)),
            nn.Conv2d(f5_r, f5, kernel_size=(1,5), padding=(0, 2)),
            nn.Conv2d(f5, f5, kernel_size=(5,1), padding=(2,0))
        )
        
        
        # 3x3 pool -> 1x1
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(x)
        
        return torch.cat([layer1, layer2, layer3, layer4], 1)
    