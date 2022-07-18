from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
import cv2 as cv
import lib_aug_scorecam as auggc
import torchvision.models as models
from utils import *

import keras
from keras.applications import VGG16

from keras_applications.vgg16 import preprocess_input
from keras_preprocessing.image import load_img, img_to_array

import sys  

def visualizations(image_name, class_index):
  model = VGG16(weights='imagenet')
  input_shape = (224, 224)
  conv_layers = model.layers[-6].output
  softmax_output = model.layers[-1].output

  source_img = load_img(image_name, target_size=input_shape)
  img = img_to_array(source_img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img, backend=keras.backend)

  # test image name
  img_name = image_name
  idx = int(class_index)

  # load the test image
  img_path = "./imgs/{}".format(img_name)
  img2 = skimage.io.imread(image_name)
  img2 = cv.resize(img2,input_shape)

  # super resolution parameters
  num_aug = 100
  learning_rate = 0.01
  lambda_eng = 0.001 * num_aug
  lambda_tv = 0.002 * num_aug
  thresh_rel = 0.15
  num_iter = 1000

  # augmentation parameters
  angle_min = -0.5  # in radians --> -28.6 degrees
  angle_max = 0.5 ##in radians --> 28.6 degrees
  angles = np.random.uniform(angle_min, angle_max, num_aug)
  shift_min = -30
  shift_max = 30
  shift = np.random.uniform(shift_min, shift_max, (num_aug, 2))
  # first Grad-CAM is not augmented
  angles[0] = 0
  shift[0] = np.array([0, 0])

  augmenter = auggc.Augmenter(num_aug, (224, 224))

  superreso = auggc.Superresolution(augmenter=augmenter, learning_rate=learning_rate, camsz=[224,224])

  #call ScoreCamModel Class __init__
  vgg_scoreCAM = auggc.ScoreCamModel(model_input=model.input, last_conv_output=conv_layers, softmax_output=softmax_output, input_shape=input_shape, input=img, class_id=idx)
  #call SingleScoreCamModel Class __init__
  Single_scoreCAM = auggc.SingleScoreCamModel(model_input=model.input, last_conv_output=conv_layers, softmax_output=softmax_output, input_shape=input_shape, input=img, class_id=idx)

  img_batch = augmenter.direct_augment(img2, angles, shift)
  idx_batch = [idx for ii in range(num_aug)]

  cams = vgg_scoreCAM.compute_cam(img_batch, idx_batch)[ :, :, :, np.newaxis]

  cam_full_tv = superreso.super_mixed(cams / np.max(cams, axis=(1, 2, 3), keepdims=True),
                                      angles, shift, lmbda_tv=lambda_tv, lmbda_eng=lambda_eng, 
                                      niter=num_iter).squeeze()

  # overlay CAMs to input image
  cam_no_blue = superreso.overlay(cam_full_tv, img2, name=image_name+"-ASC")

  def draw_original_and_heatmap():
      heatmap = Single_scoreCAM.final_class_activation_map
      plt.figure(figsize=(7,7))
      plt.imshow(source_img, alpha=0.5)
      plt.imshow(heatmap, cmap='plasma', alpha=0.5)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                          hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
      plt.savefig(image_name+"-SSC.png", dpi=50)
      
      
  draw_original_and_heatmap()
  
if __name__ == '__main__':
  image_name,class_index = sys.argv[1:3]
  visualizations(image_name, class_index)
