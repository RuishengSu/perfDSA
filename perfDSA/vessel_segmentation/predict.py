import logging
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def predict_segmentation(net, image_array, device=torch.device('cpu')):
    """
    Predicts the segmentation of an image using a given neural network.

    Args:
        net (torch.nn.Module): The neural network model used for segmentation.
        image_array (numpy.ndarray): The input image to be segmented, represented as a numpy array.
        device (torch.device, optional): The device on which the segmentation is performed. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: The predicted segmentation of the input image.
    """
    net.eval()

    image = transforms.ToPILImage()(image_array)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device=device)

    with torch.no_grad():
        segmentation = net(image)

    return segmentation.argmax(dim=1)[0].cpu().detach().numpy().astype(np.uint8)
