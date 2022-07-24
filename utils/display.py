import torchvision
from PIL import Image, ImageDraw
from torchsummary import summary

def display_bboxes(image, y_pred, y_true):
    """
    Displays bboxes on an image
    """

    image = torchvision.transforms.functional.to_pil_image(image)
    img_draw = ImageDraw.Draw(image)  

    img_width, img_height = image.size
    cells = y_true.shape[1]
    cell_size = img_width/cells
    rel_cell_size = 1/cells

    for y_cell in range(y_true.shape[0]):
        for x_cell in range(y_true.shape[1]):

            current_cell = y_true[x_cell, y_cell]


            if current_cell[0,4] > 0.1:
                x  = x_cell * rel_cell_size + current_cell[0, 0]
                y  = y_cell * rel_cell_size + current_cell[0, 1]


                bbox_coords = (
                    (x - current_cell[0, 2]/2 ) *img_height,
                    (y - current_cell[0, 3]/2 ) *img_height,
                    (x + current_cell[0, 2]/2 ) *img_height,
                    (y + current_cell[0, 3]/2 ) *img_height

                )
                img_draw.rectangle(bbox_coords, width=3, outline="blue")

    return img_draw._image
