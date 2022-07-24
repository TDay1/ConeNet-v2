from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import os
from PIL import Image, ImageDraw



class ConeSet(Dataset):
    """
    Implementing YOLOV1 style dataset
    """
    def __init__(self, root_dir, offset=0, device="cpu", size=32):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        
        self.size = size
        self.root_dir = root_dir
        self.device = device
        self.offset = offset
        
        # Grid cells in each direction (total = s^2)
        self.S = 10
        self.B = 2

        self.resolution = 624

        #self.dtype = torch.cuda.HalfTensor

        self.dataset = []
        self.__build_dataset__()

    def __build_dataset__(self):
        """
        build the dataset
        """
        for i in range(self.offset, self.offset + self.size):

            exists = os.path.isfile(os.path.join(self.root_dir,
                            (str(i) + '.txt')))
            if exists  == 0:
                image = self.load_image(i)
                label = self.load_label(i)
                self.dataset.append((image, label))

            if i % 100 == 0:
                print(f'Loaded {i/(self.offset + self.size)}')
    

    
    def __len__(self):
            return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def load_image(self, index):
        """
        Load image and label from the dataset
        """
        filename = os.path.join(self.root_dir,
                            (str(index) + '.PNG'))


        image = Image.open(filename)

        image = torchvision.transforms.functional.resize(image, (self.resolution, self.resolution))
        image = torchvision.transforms.functional.pil_to_tensor(image)
        image = image / 255

        return image.to(self.device)

    def load_label(self, index):
        """
        Load label from the dataset
        """

        # read file
        filename = os.path.join(self.root_dir,
                            (str(index) + '.txt'))

        labels = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                labels.append([float(x) for x in line])


        # Convert
        
        Y = torch.zeros((self.S, self.S, self.B, 5))

        # Split into grid cells
        cell_wdth = 1/self.S

        for bbox in labels:
            # the grid cell in which to place the box
            grid_cell = (int(bbox[1] // cell_wdth), int(bbox[2] // cell_wdth))
            # The centre coords of the box relative to the cell
            grid_relative_coords = (bbox[1] % cell_wdth, bbox[2] % cell_wdth)

            # Put box in nxt available slot in the grid cell
            for i in range(self.B):
                if Y[grid_cell[0], grid_cell[1], i, 0] == 0:
                    Y[grid_cell[0], grid_cell[1], i, 0]= grid_relative_coords[0]
                    Y[grid_cell[0], grid_cell[1], i, 1] = grid_relative_coords[1]
                    Y[grid_cell[0], grid_cell[1], i, 2] = bbox[3]
                    Y[grid_cell[0], grid_cell[1], i, 3] = bbox[4]
                    Y[grid_cell[0], grid_cell[1], i, 4] = 1
                    break

        #Convert Y to required device
        Y = Y.to(self.device)

        return Y