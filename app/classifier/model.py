import torch.nn as nn
import torch
from lightning.pytorch import LightningModule,seed_everything,Trainer
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau


import numpy as np

import pytorchvideo
import lightning.pytorch as pl
import pytorchvideo.models.hub.slowfast as slowfast
from lion_pytorch import Lion
import pytorchvideo.models.hub.x3d as x3d
import os
from torch.utils.data import DataLoader
NUM_WORKERS =  os.cpu_count() or 0
from pytorchvideo.data import LabeledVideoDataset,Kinetics, make_clip_sampler,labeled_video_dataset

from pytorchvideo.transforms import(
    ApplyTransformToKey,Normalize,RandomShortSideScale,Permute,UniformTemporalSubsample,  ShortSideScale
)

from torchvision.transforms import(
    Compose,Lambda,RandomCrop,RandomHorizontalFlip,Resize,
)
from torchvision.transforms._transforms_video import(
    CenterCropVideo,
    NormalizeVideo
)
class videoClassifer(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.prepresentation=slowfast.slowfast_r50(pretrained=True)
        self.prepresentation.blocks[6].proj=nn.Linear(in_features=2304, out_features=1000, bias=True)
        self.fc=nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500,out_features=400),
            nn.ReLU(),
            
            nn.Linear(in_features=400,out_features=1)
        )
        for param in self.prepresentation .blocks[:] .parameters():
            param.requires_grad = False

    def forward(self,x):
        x=self.prepresentation(x)
        x=self.fc(x)

        return x
    
    
side_size = 300
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3
clip_duration = (num_frames * sampling_rate)/frames_per_second

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


    
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)
def load_model(path=r"weight/VideoClassifierSlowFast_50_size_300.ckpt"):
    device= "cuda:0"
    _model=videoClassifer.load_from_checkpoint(path,map_location=device)

    return _model


# if __name__=="__main__":
#     model=load_model()
#     input_sample =[ torch.randn((1, 3, 8, 300, 300))]
#     # filepath = "model.onnx"
#     # model.to_onnx(filepath,input_sample, export_params=True)
#     print(model(input_sample))






