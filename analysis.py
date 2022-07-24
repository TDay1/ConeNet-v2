from numpy import number
import torch
from torch.utils.data import DataLoader
import os.path
import time

from utils.dataset import ConeSet
from utils.model import ConeNet
from utils.display import display_bboxes


def benchmark(net, dataset_iter):

    # Benchmark how long it takes to run on CPU
    start = time.time()

    for data in dataset_iter:
        net(data[0])

    end = time.time()

    print(f"Processed {number_of_images} image(s) in {round(end - start, 4)}s    ({number_of_images / round(end - start, 4)} fps)")

def visualise(net, dataset_iter):
    start = time.time()

    for index, data in enumerate(dataset_iter):
        out = net(data[0])
        print(data[0][0].shape)
        annotated_image = display_bboxes(data[0][0], None, out[0])
        annotated_image.save('ConeNetv2-' + str(index) + '.png', "PNG")

    end = time.time()

    print(f"Processed, visualised, and saved {number_of_images} image(s) in {round(end - start, 4)}s    ({number_of_images / round(end - start, 4)} fps)")



# Define loader
number_of_images = 12
dataset_start_offset = 50
dataset = ConeSet('C:\\Users\\tday\\datasets\\racing\\ConeSet-v2\\train', offset=dataset_start_offset, device="cpu", size=number_of_images)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
load_iter = iter(loader)

# Load net
net = ConeNet()
# ConeNet stored as Cuda
net.to('cuda')
# Load checkpoint
checkpoint = torch.load(f'C:\\Users\\tday\\code\\racing\\checkpoints\\1625.pt')
net.load_state_dict(checkpoint['model_state_dict'])
# Move to CPU for analysis
net.to('cpu')

visualise(net, load_iter)
















#for data in load_iter:
#    out_data = net(data[0])
#    display_bboxes(data[0][0], None, out_data[0])
#end = time.time()
#print(end - start)

#display_bboxes(data[0][0], None, out_data[0])