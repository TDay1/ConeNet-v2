from numpy import number
import torch
from torch.utils.data import DataLoader
import os.path
import time
import cv2
import numpy as np

from utils.dataset import ConeSet
from utils.model import ConeNet
from utils.display import display_bboxes

import torchvision

from PIL import Image


def infer():

    # Setup nerual net
    net = ConeNet()
    # ConeNet stored as Cuda
    net.to('cuda')
    # Load checkpoint
    checkpoint = torch.load(f'C:\\Users\\tday\\code\\racing\\checkpoints\\1625.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    # Move to CPU for analysis
    net.to('cpu')



    vidcap = cv2.VideoCapture('./samples/test_clip2.mp4')
    success,image = vidcap.read()
    count = 0
    
    writer = cv2.VideoWriter('./samples/test_output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (624, 624))

    while success:
        # Prep image
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame)
        frame = torchvision.transforms.functional.resize(im_pil, (624, 624))
        tensor = torchvision.transforms.functional.pil_to_tensor(frame)
        tensor = tensor / 255
        tensor = torch.unsqueeze(tensor, 0)

        # Inference
        net_out = net(tensor)

        # Visualise
        bbox_image = display_bboxes(tensor[0], None, net_out[0], conf_thresh=0.2)

        # save
        image_out = cv2.cvtColor( np.array(bbox_image), cv2.COLOR_RGB2BGR)
        writer.write( image_out )

        # Log to console
        count += 1
        if count % 100 == 0:
            print(count)

        # Get next frame
        success,image = vidcap.read()

    # Release output
    writer.release()
    
infer()