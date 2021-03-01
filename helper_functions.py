import numpy as np
import torch
import os
from torchvision import datasets, transforms
from PIL import Image
from my_model import MyModel

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]


def get_training_transformations():
    t = transforms.Compose([transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(image_net_mean, image_net_std)])
    return t


def get_testing_transformations():
    t = transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(image_net_mean, image_net_std)])
    return t


def get_data_loader(directory, tforms, batch_size=64, shuffle=False):
    image_data = datasets.ImageFolder(directory, transform=tforms)
    data_loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle)

    return image_data, data_loader


def process_image(filename):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns pytorch tensor
    '''
    image = Image.open(filename)

    # resize image, with shortest side = 256
    new_width = new_height = 256
    if image.width < image.height:
        new_width, new_height = 256, image.height
    else:
        new_width, new_height = image.width, 256

    # resize to 256
    image.thumbnail((new_width, new_height))

    # center crop image
    left = (image.width / 2) - (224 / 2)
    right = (image.width / 2) + (224 / 2)
    top = (image.height / 2) - (224 / 2)
    bottom = (image.height / 2) + (224 / 2)

    image = image.crop((left, top, right, bottom))

    mean = np.array(image_net_mean)
    std = np.array(image_net_std)

    # update color channel to float between 0-1
    im_array = np.array(image) / 255
    im_array = (im_array - mean) / std

    # make color channel first dimension
    im_array = im_array.transpose(2, 0, 1)

    # convert to pytorch
    im_array = torch.from_numpy(im_array).type(torch.FloatTensor)

    return im_array


def load_model_from_checkpoint(chkpt_path, gpu=False):
    arch = os.path.basename(chkpt_path).split('_')[0]
    saved_on_device = os.path.basename(chkpt_path).split('_')[1]
    device = torch.device("cuda" if gpu else "cpu")
    if saved_on_device == 'cuda' and gpu or saved_on_device == 'cpu' and not gpu:
        checkpoint = torch.load(chkpt_path)
    else:
        checkpoint = torch.load(chkpt_path, map_location=device.type)

    hidden_layers = checkpoint['hidden_layers']

    model = MyModel(arch=arch, hidden_layers=hidden_layers)
    model = model.get_model()
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model