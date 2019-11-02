import torch
import torch.nn as nn
import torchvision.models as models

# get the layers in resnet18 but does not have the part of fc

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  

        # no need the fully connected layer

        ''' declare layers used in this network'''
        # resnet18
        # self.resnet18 = resnet18() # Nx3x352x448 -> Nx512x11x14
        
        # ResNet(
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18short = nn.Sequential(*(list(resnet18.children())[:-2]))

        # torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        # transpose conv1 relu
        # Nx512x11x14 -> Nx256x22x28
        self.tconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        
        # transpose conv2 relu
        # Nx256x22x28 -> Nx128x44x56
        self.tconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        
        # transpose conv3 relu
        # Nx128x44x56 -> Nx64x88x112
        self.tconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        
        # transpose conv4 relu
        # Nx64x88x112 -> Nx32x176x224
        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        
        # transpose conv5 relu
        # Nx32x176x224 -> Nx16x352x448
        self.tconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        
        # convolution
        # Nx16x352x448 -> Nx9x352x448
        self.conv = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)
        
        # argmax
        # Nx9x352x448 -> Nx352x448
        #self.argmax = torch.argmax(self, )
        
        '''
        # first block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # 64x64 -> 64x64
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # second block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
        # third block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        '''
        
        '''
        # classification
        # self.avgpool = nn.AvgPool2d(16)
        # self.fc = nn.Linear(64, 4)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128, 4)
        '''

    def forward(self, img):
        
        # Resnet
        x = self.resnet18short(img)
        # PDF working
        x = self.relu1(x)
        x = self.tconv1(x)
        x = self.relu2(x)
        x = self.tconv2(x)
        x = self.relu3(x)
        x = self.tconv3(x)
        x = self.relu4(x)
        x = self.tconv4(x)
        x = self.relu5(x)
        x = self.tconv5(x)
        x = self.conv(x)

        '''
        x = self.relu1(self.bn1(self.conv1(img)))
        x = self.maxpool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        x = self.avgpool(x).view(x.size(0),-1)
        x = self.fc(x)
        '''
        return x

'''
        self.rconv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rbn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rrelu = nn.ReLU(inplace=True)
        self.rmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # (layer1): Sequential(
        # (0): BasicBlock
        self.rs1b0conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b0bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs1b0relu = nn.ReLU(inplace=True)
        self.rs1b0conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b0bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # (1): BasicBlock(
        self.rs1b1conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b1bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs1b1relu = nn.ReLU(inplace=True)
        self.rs1b1conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b1bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
  
        # (layer2): Sequential(
        # (0): BasicBlock(
        self.rs2b0conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs2b0bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs2b0relu = nn.ReLU(inplace=True)
        self.rs2b0conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b0bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (downsample): Sequential(
        self.rs2b0downsample0 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs2b0downsample1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (1): BasicBlock(
        self.rs2b1conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b1bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs2b1relu = nn.ReLU(inplace=True)
        self.rs2b1conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b1bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (layer3): Sequential(
        # (0): BasicBlock(
        self.rs3b0conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs3b0bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs3b0relu = nn.ReLU(inplace=True)
        self.rs3b0conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b0bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (downsample): Sequential(
        self.rs3b0downsample0 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs3b0downsample1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
      
        # (1): BasicBlock(
        self.rs3b1conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b1bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs3b1relu = nn.ReLU(inplace=True)
        self.rs3b1conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b1bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (layer4): Sequential(
        # (0): BasicBlock(
        self.rs4b0conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs4b0bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs4b0relu = nn.ReLU(inplace=True)
        self.rs4b0conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b0bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (downsample): Sequential(
        self.rs4b0downsample0 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs4b0downsample1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
      
        # (1): BasicBlock(
        self.rs4b1conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b1bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs4b1relu = nn.ReLU(inplace=True)
        self.rs4b1conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b1bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

'''


'''
        x = self.rrelu(self.rbn1(self.rconv1(img)))
        x = self.rmaxpool(x)
        # (1)(0): BasicBlock
        x = self.rs1b0bn2(self.rs1b0conv2(self.rs1b0relu(self.rs1b0bn1(self.rs1b0conv1(x)))))
        # (1)(1):
        x = self.rs1b1bn2(self.rs1b1conv2(self.rs1b1relu(self.rs1b1bn1(self.rs1b1conv1(x)))))

        # (2)(0):
        x = self.rs2b0bn2(self.rs2b0conv2(self.rs2b0relu(self.rs2b0bn1(self.rs2b0conv1(x)))))
        # (2)(d):
        x = self.rs2b0downsample1(self.rs2b0downsample0(x))
        # (2)(1):
        x = self.rs2b1bn2(self.rs2b1conv2(self.rs2b1relu(self.rs2b1bn1(self.rs2b1conv1(x)))))
        # (3)(0):
        x = self.rs3b0bn2(self.rs3b0conv2(self.rs3b0relu(self.rs3b0bn1(self.rs3b0conv1(x)))))
        # (3)(d):
        x = self.rs3b0downsample1(self.rs3b0downsample0(x))
        # (3)(1):
        x = self.rs3b1bn2(self.rs3b1(self.rs3b1conv2(self.rs3b1relu(self.rs3b1bn1(self.rs3b1conv1(x))))))
        # (4)(0):
        x = self.rs4b0bn2(self.rs4b0conv2(self.rs4b0relu(self.rs4b0bn1(self.rs4b0conv1(x)))))
        # (4)(d):
        x = self.rs4b0downsample1(self.rs4b0downsample0(x))
        # (4)(1):
        x = self.rs4b1bn2(self.rs4b1conv2(self.rs4b1relu(self.rs4b1bn1(self.rs4b1conv1(x)))))

'''

'''
# ResNet(
        self.rconv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rbn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rrelu = nn.ReLU(inplace=True)
        self.rmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # (layer1): Sequential(
        # (0): BasicBlock
        self.rs1b0conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b0bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs1b0relu = nn.ReLU(inplace=True)
        self.rs1b0conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b0bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        # (1): BasicBlock(
        self.rs1b1conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b1bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs1b1relu = nn.ReLU(inplace=True)
        self.rs1b1conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs1b1bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
  
        # (layer2): Sequential(
        # (0): BasicBlock(
        self.rs2b0conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs2b0bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs2b0relu = nn.ReLU(inplace=True)
        self.rs2b0conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b0bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (downsample): Sequential(
        self.rs2b0downsample0 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs2b0downsample1 =  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (1): BasicBlock(
        self.rs2b1conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b1bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs2b1relu = nn.ReLU(inplace=True)
        self.rs2b1conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs2b1bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (layer3): Sequential(
        # (0): BasicBlock(
        self.rs3b0conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs3b0bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs3b0relu = nn.ReLU(inplace=True)
        self.rs3b0conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b0bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (downsample): Sequential(
        self.rs3b0downsample0 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs3b0downsample1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
      
        # (1): BasicBlock(
        self.rs3b1conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b1bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs3b1relu = nn.ReLU(inplace=True)
        self.rs3b1conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs3b1bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    
        # (layer4): Sequential(
        # (0): BasicBlock(
        self.rs4b0conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.rs4b0bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs4b0relu = nn.ReLU(inplace=True)
        self.rs4b0conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b0bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (downsample): Sequential(
        self.rs4b0downsample0 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.rs4b0downsample1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
      
        # (1): BasicBlock(
        self.rs4b1conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b1bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.rs4b1relu = nn.ReLU(inplace=True)
        self.rs4b1conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.rs4b1bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
'''
