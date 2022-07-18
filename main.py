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
      
      foreground = np.uint8(heatmap / heatmap.max() * 255)
      foreground = cv.applyColorMap(foreground, cv.COLORMAP_JET)
      BLUE_MIN = np.array([75, 50, 50],np.uint8)
      BLUE_MAX = np.array([130, 255, 255],np.uint8)
      hsv_img = cv.cvtColor(foreground,cv.COLOR_BGR2HSV)
      frame_threshed = cv.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
      frame_threshed = cv.bitwise_and(foreground, foreground, mask=frame_threshed)
      frame_threshed = np.array(frame_threshed , dtype=np.uint8)
      subtract_im = cv.subtract(foreground, frame_threshed)
      image2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
      Z = cv.addWeighted(image2, 0.6, subtract_im, 0.4, 0)
      cv.imwrite(image_name+"-SSC-FILTER.png", Z)
      #********************** Overlay image and blackout background ******************
      blank_image = np.zeros((224,224,3), np.uint8)
      Y1 = cv.addWeighted(blank_image, 0.5, subtract_im, 0.5, 0)
      edges = cv.Canny(Y1,60,200)
      kernel = np.ones((2,2),np.uint8)
      dilation = cv.dilate(edges,kernel,iterations = 10)
      closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
      #cv.imwrite("SSC edges-before.png", edges)
      cnts = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      print("Number of SSC Contours is: " + str(len(cnts)))
      for c in cnts:
         cv.drawContours(closing,[c], -1, (255,255,255), -1)
      #cv.imwrite("SSC edges-after.png", edges)
      result = cv.bitwise_and(image2, image2, mask= closing)
      cv.imwrite(image_name+ "-SSC-BLACKOUT.png", result)

  draw_original_and_heatmap()
  
  #************************Generate Bounding Boxes for ASC heatmap********************#
  bbox_tv = auggc.get_bbox_from_cam(np.float32(cam_no_blue))
  rect_tv = auggc.get_rectangle(bbox_tv, color="red")

  fig,ax = plt.subplots(1)
  ax.imshow(img2)
  ax.add_patch(rect_tv)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                      hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(image_name+"-ASC-RECT.png", dpi=50)
  #************************Generate Bounding Boxes for SSC heatmap********************#
  SSC_cam = Single_scoreCAM.final_class_activation_map
  SSC_bbox_tv = auggc.get_bbox_from_cam(SSC_cam)
  SSC_rect_tv = auggc.get_rectangle(SSC_bbox_tv, color="red")
  fig,ax = plt.subplots(1)
  ax.imshow(img2)
  ax.add_patch(SSC_rect_tv)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                      hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(image_name+"-SSC-RECT.png", dpi=50)
  
if __name__ == '__main__':
  image_name,class_index = sys.argv[1:3]
  visualizations(image_name, class_index)
