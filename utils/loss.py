import torch

class Yoloss(torch.nn.Module):
    def __init__(self):
        """
        (modified) Yolo loss function.
        3 Losses:
        1. confidence loss (C) 
        2. coordinate loss (Co) 
        3. class loss (Ce) - Always 0 as we only have one class
        """

        super(Yoloss, self).__init__()

        self.S=10
        self.B=2

        self.lambda_coord = 5

    def forward(self, y_true, y_pred):
        conf_loss = 0
        bbox_loss = 0

        # BBOX predicition
        pred_conf = y_pred[:, :, :, :, 4].flatten()
        truth_conf = y_true[:, :, :, :, 4].flatten()
        # sigma (c - c)^2
        conf_loss += torch.square( pred_conf - truth_conf ).sum()
        
        # BBOX loss
        pred_x = y_pred[:, :, :, :, 0].flatten()
        truth_x = y_true[:, :, :, :, 0].flatten()

        pred_y = y_pred[:, :, :, :, 1].flatten()
        truth_y = y_true[:, :, :, :, 1].flatten()

        # sigma ((x-x)^2 + (y-y)^2)
        bbox_loss += torch.square( pred_x - truth_x ).sum() + torch.square( pred_y - truth_y ).sum()

        pred_w = y_pred[:, :, :, :, 2].flatten()
        truth_w = y_true[:, :, :, :, 2].flatten()

        pred_h = y_pred[:, :, :, :, 3].flatten()
        truth_h = y_true[:, :, :, :, 3].flatten()

        # sigma ((w-w)^2 + (h-h)^2)
        bbox_loss += self.lambda_coord * ( torch.square( pred_w - truth_w ).sum() + torch.square( pred_h - truth_h ).sum() )

        return conf_loss + bbox_loss

