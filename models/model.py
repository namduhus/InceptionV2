import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Modules 가져오기
from modules.inception_V2_A import InceptionV2_A  # Figure 5
from modules.inception_V2_B import InceptionV2_B  # Figure 6
from modules.inception_V2_C import InceptionV2_C  # Figure 7


class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV2, self).__init__()
        
        
        # 초기 conv 계층
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)            # 299x299x3 -> 149x149x32
        self.conv2 = nn.Conv2d(32,32, kernel_size=3)                      # 149x149x32 -> 147x147x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)          # 147x147x32 -> 147x147x64
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # 147x147x64 -> 73x73x64
        
        
        # 추가 conv 계층
        self.conv4 = nn.Conv2d(64, 80, kernel_size=3)                      # 73x73x64 -> 71x71x80
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=2)           # 71x71x80 -> 35x35x192
        self.conv6 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        
        self.channel_adjust = nn.Conv2d(256, 288, kernel_size=1)

        
        # 35x35 InceptionA 3회실시 (Figure 5)
        self.inception_A1 = InceptionV2_A(288, 64, 96, 128, 16, 48, 48)
        self.inception_A2 = InceptionV2_A(288, 64, 96, 128, 16, 48, 48)
        self.inception_A3 = InceptionV2_A(288, 64, 96, 128, 16, 48, 48)
        
        
        # 35x35 -> 17x17
        self.reduction_1 = nn.Sequential(
            nn.Conv2d(288, 768, kernel_size=3, stride=2, padding=1)
        )  # 35x35x288 -> 17x17x768
        
        
        # 17x17 InceptionB 5회 실시 (Figure 6)
        self.inception_B1 = InceptionV2_B(768, 192, 256, 320, 64, 128, 128)
        self.inception_B2 = InceptionV2_B(768, 192, 256, 320, 64, 128, 128)
        self.inception_B3 = InceptionV2_B(768, 192, 256, 320, 64, 128, 128)
        self.inception_B4 = InceptionV2_B(768, 192, 256, 320, 64, 128, 128)
        self.inception_B5 = InceptionV2_B(768, 192, 256, 320, 64, 128, 128)

        
        
        # 17x17 -> 8x8 그리드 축소 (Figure 10)
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(768, 1280, kernel_size=3, stride=2, padding=0)        # 17x17x768 -> 8x8x1280
        )
        
        
        # 8x8 Inception 2회 실시 (Figure 7)
        self.inception_C1 = InceptionV2_C(1280, 512, 512, 384, 512, 512, 640)
        self.inception_C2 = InceptionV2_C(2048, 512, 512, 384, 512, 512, 640)




        
        
        # 최종 풀링 및 분류 계층
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))                            # 8x8x2048 -> 1x1x2048
        self.fc = nn.Linear(2048, num_classes)                              # 1x1x2048 -> 1x1x1000 
        
        
    def forward(self, x):
      # 초기 Conv 및 Pooling 계층
      x = self.conv1(x)

      x = self.conv2(x)
      
      x = self.conv3(x)

      x = self.pool1(x)
      
      x = self.conv4(x)
      
      x = self.conv5(x)
       
      x = self.conv6(x)

      x = self.channel_adjust(x)

      # Inception A 모듈 (3회)
      x = self.inception_A1(x)
      
      x = self.inception_A2(x)
      
      x = self.inception_A3(x)

      # 35x35 -> 17x17 그리드 축소
      x = self.reduction_1(x)
      
      # Inception B 모듈 (5회)
      x = self.inception_B1(x)
      x = self.inception_B2(x)
      x = self.inception_B3(x)
      x = self.inception_B4(x)
      x = self.inception_B5(x)

      # 17x17 -> 8x8 그리드 축소
      x = self.reduction_2(x)
      
      # Inception C 모듈 (2회)
      x = self.inception_C1(x)
      x = self.inception_C2(x)

      # 최종 풀링 및 분류 계층
      x = self.pool2(x)

      x = torch.flatten(x, 1)
      
      x = self.fc(x)

      return x