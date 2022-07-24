from unicodedata import name
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import os.path
import torch.optim as optim
import time

from utils.dataset import ConeSet
from utils.loss import Yoloss
from utils.model import ConeNet
import wandb

def train():



    # init wandb
    wandb.init(project="ConeNet-v2", entity="tomd")

    wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 1500,
    "batch_size": 24,
    }


    # Define loader
    dataset = ConeSet('C:\\Users\\tday\\datasets\\racing\\ConeSet-v2\\train', offset=20, device="cpu", size=3000)
    
    loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,
        collate_fn=None,
        pin_memory=True,
        )

    #Define models

    # Start from checkpoint
    net = ConeNet()
    net.to('cuda')

    # Optimiser
    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    checkpoint = torch.load(f'C:\\Users\\tday\\code\\racing\\checkpoints\\150.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    

        # Define loss
    yoloss = Yoloss()

    epochs = 1500
    resume_point = 150
    for epoch in range(resume_point, resume_point+epochs):
        epoch_loss = []

        start_time = time.time()

        for batch, data in enumerate(loader, 0):


            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')


            # Zdero optim grads
            optimizer.zero_grad()

            # forward + backward + optimize + L + ratio
            outputs = net(inputs)

            batch_loss = yoloss(labels, outputs)
            batch_loss.backward()
            
            optimizer.step()

            epoch_loss.append( batch_loss.item() )

        end_time = time.time()    
        print(f'Epoch {epoch}: loss={sum(epoch_loss)}')

        wandb.log({
            "loss": sum(epoch_loss),
            "loss-per-img": sum(epoch_loss)/(2930),
            "time-per-epoch": start_time-end_time
            })

        # Checkpoint
        if epoch % 25 == 0:

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': sum(epoch_loss),
                        }, f'C:\\Users\\tday\\code\\racing\\checkpoints\\{epoch}.pt')



if __name__ == "__main__":
    train()