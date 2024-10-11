GOOGLE_DRIVE_PATH = "D://AppData//AI//eecs498-007//A6"
import sys
sys.path.append(GOOGLE_DRIVE_PATH)

import time, os
os.environ["TZ"] = "US/Eastern"

import os
import torch
import torchvision
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL
from style_transfer import *
from a6_helper import *
from eecs598.grad import rel_error

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
dtype = torch.float
device = 'cuda'
to_float_cuda = torch.cuda.FloatTensor

cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(to_float_cuda)

for param in cnn.parameters():
    param.requires_grad = False

check_scipy()
answers = dict(np.load('style-transfer-checks.npz'))

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def features_from_img(imgpath, imgsize):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(to_float_cuda)
    return extract_features(img_var, cnn), img_var


def get_image_masks(img_paths, img_size):
    masks = []
    for path in img_paths:
        masks.append(
            get_zero_one_masks(PIL.Image.open(path), img_size).type(to_float_cuda)
        )
    return torch.cat(masks, 0)


def extract_layer_masks(features, masks):
    layer_masks = []

    for feat in features:
        feat_height, feat_width = feat.shape[-2], feat.shape[-1]
        feat_transform = T.Resize((feat_height, feat_width))
        feat_masks = feat_transform(masks)
        layer_masks.append(feat_masks)

    return layer_masks


def extract_regional_features(x, cnn, R=2):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.
    - R: stack the features R times to generate the image region guidance

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, R, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat.unsqueeze(1).repeat(1, R, 1, 1, 1))
        prev_feat = next_feat
    return features


def guided_style_transfer(content_image, style_images, image_size, style_size,
                          content_layer, content_weight, content_masks,
                          style_layers, style_weights, style_masks, tv_weight,
                          init_random=False, save_image=False, result_filename=None):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_images: list of filenames of style images
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - content_masks: binary masks of the content image for the corresponding regions
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - style_masks: binary masks of the sylte image for the corresponding regions
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    - save_image: boolean flag for saving the image
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size).type(to_float_cuda)
    feats = extract_regional_features(content_img, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style images
    # For this simplified assignment problem, we use 2 style images
    style1_img, style2_img = [
        preprocess(PIL.Image.open(style_image), size=style_size).type(to_float_cuda)
        for style_image in style_images
    ]
    style1_feats = extract_features(style1_img, cnn)
    style2_feats = extract_features(style2_img, cnn)

    # generate content mask for each shape of the style features
    content_masks = extract_layer_masks(style1_feats, content_masks)

    # Stack style features
    feats = []
    for i in range(len(style1_feats)):
        style_feats = torch.stack((style1_feats[i], style2_feats[i]), dim=1)
        feats.append(style_feats)
    style_masks = extract_layer_masks(feats, style_masks)

    style_targets = []
    for idx in style_layers:
        style_targets.append(guided_gram_matrix(feats[idx].clone(), style_masks[idx]))

    # Initialize output image to content image or noise
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(to_float_cuda)
    else:
        img = content_img.clone().type(to_float_cuda)

    # We do want the gradient computed on our image!
    img.requires_grad_()

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img], lr=initial_lr)

    f, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style1 Source Img.')
    axarr[2].set_title('Style2 Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style1_img.cpu()))
    axarr[2].imshow(deprocess(style2_img.cpu()))
    plt.show()
    plt.figure()

    for t in range(200):
        if t < 190:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_regional_features(img, cnn)

        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks)
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss

        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            plt.imshow(deprocess(img.data.cpu()))
            plt.show()
    print('Iteration {}'.format(t))
    plt.axis('off')
    plt.imshow(deprocess(img.data.cpu()))
    if save_image:
        plt.savefig(os.path.join(GOOGLE_DRIVE_PATH, result_filename))
    plt.show()

image_folder_path = os.path.join(GOOGLE_DRIVE_PATH, 'images')
image_size = (192, 288)
content_sky_mask_path = os.path.join(image_folder_path, 'fig2_content_sky.jpg')
content_nosky_mask_path = os.path.join(image_folder_path, 'fig2_content_nosky.jpg')
content_masks = get_image_masks([content_nosky_mask_path, content_sky_mask_path], image_size).unsqueeze(0)
style1_nosky_mask_path = os.path.join(image_folder_path, 'fig2_style1_nosky.jpg')
style2_sky_mask_path = os.path.join(image_folder_path, 'fig2_style2_sky.jpg')
style_masks = get_image_masks([style1_nosky_mask_path, style2_sky_mask_path], image_size).unsqueeze(0)

content_image_path = os.path.join(image_folder_path, 'fig2_content.jpg')
style1_image_path = os.path.join(image_folder_path, 'fig2_style1.jpg')
style2_image_path = os.path.join(image_folder_path, 'fig2_style2.jpg')

# Spatial Style Transfer
params_inv = {
    'content_image' : content_image_path,
    'style_images' : [style1_image_path, style2_image_path],
    'image_size' : (192, 288),
    'style_size' : (192, 288),
    'content_layer' : 3,
    'content_weight': 2e-2,
    'content_masks': content_masks,
    'style_layers' : [1, 4, 6, 7],
    'style_weights' : [300000, 1000, 15, 3],
    # 'style_weights': [0, 0, 0, 0],
    'style_masks': style_masks,
    'tv_weight' : 2e-2,
    'init_random': True, # we want to initialize our image to be random
    'save_image' : True,
    'result_filename': 'spatial_style_transfer.jpg'
}

guided_style_transfer(**params_inv)
