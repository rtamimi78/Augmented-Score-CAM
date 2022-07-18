import cv2
import matplotlib.patches as patches
import numpy as np
import skimage.io
import skimage.transform
import skimage.measure
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from keras import Model
import numpy as np

from scam.exceptions import InvalidState
from scam.utils import resize_activations, normalize_activations

sess = tf.InteractiveSession(graph=tf.Graph())

def get_rectangle(bbox, color='r'):
    """ Get a rectangle patch from `bbox` for plotting purposes.
    :param bbox: input bounding box
    :param color: rectangle color
    :return rect: a colored rectangle patch
    """
    # Create a Rectangle patch
    (xmin, xmax, ymin, ymax) = bbox
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor="none")
    return rect

def order_bbox_from_sk(bbox):
    return bbox[1], bbox[3], bbox[0], bbox[2]

def get_bbox_from_cam(cam_full, thresh_rel=0.15):
    """ Compute the bounding box from `cam_full` according to `thresh_rel`.
    Fit a box around `cam_full` where its values are greater than the 15% of its maximum.
    :param cam_full: the input CAM
    :param thresh_rel: the threshold values (same as in https://arxiv.org/abs/1610.02391)
    :return: the bounding box for `cam_full`
    """
    thresh = thresh_rel * np.max(cam_full)
    mask = cam_full > thresh

    label_mask = skimage.measure.label(mask)
    all_regions = skimage.measure.regionprops(label_mask)

    all_areas = [region["area"] for region in all_regions]
    idx = np.argmax(all_areas)
    try:
        idx = idx[0]
    except:
        pass
    region = all_regions[idx]
    bbox = region.bbox

    return order_bbox_from_sk(bbox)

def bb_intersection_over_union(boxA, boxB):
    """ Compute the intersection over union (IOU) metrics between two bounding boxes.
    :param boxA: bounding box 1
    :param boxB: bounding box 2
    :return iou: the intersection over union of `boxA` and `boxB`
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class ScoreCamModel:
    def __init__(self, model_input, last_conv_output, softmax_output, input_shape, input, class_id, cam_batch_size=None):
        """
        Prepares class activation mappings
        :param model_input: input layer of CNN, normally takes batch of images as an input. Currently batch must be limited to a single image
        :param last_conv_output: last convolutional layer. The last conv layer contains the most complete information about image.
        :param softmax_output: flat softmax (or similar) layer describing the class certainty
        :param input_shape: Expecting a batch of a single input sample 1 x M X N X ...; it is assumed that 2D image of M x N dimensions is served as an input, which can be multiplied with a 2D-mask.
        :param cam_batch_size: Optional, defaults to None, which will result in inference of batches of size 32.
        """
        self.model_input = model_input
        self.last_conv_output = last_conv_output
        self.softmax_output = softmax_output
        self.last_conv_model = Model(inputs=model_input, outputs=last_conv_output)
        self.softmax_model = Model(inputs=model_input, outputs=softmax_output)
        self.input_shape = input_shape
        self.cam_batch_size = cam_batch_size
        self.input = input
        self.class_id = class_id
        self.normalized_maps = None
        self.classes_activation_scale = None

        graph = tf.get_default_graph()
        tns_input = graph.get_tensor_by_name("input_1:0")
        
        # input index for the target class
        tns_index = tf.placeholder("int32", [None])
        self.tns_index = tns_index
        self.tns_input = tns_input
        
    def preprocess_input(self, image_batch):
        """ Pre-process `image_batch` to meet the specifications of the loaded model.
        :param image_batch: the input images
        :return all_img_pre: the pre-processed input images
        """
        all_img_pre = [keras.applications.vgg16.preprocess_input(image.copy())[np.newaxis, :, :, :] for image in image_batch]
        
        return all_img_pre

    def compute_full_prob(self, image_batch):
        """ Classify the images and return the probability values over the considered label space.
        :param image_batch: the input images
        :return full_prob: the probailities values for the predicted classes
        """
        all_img_pre = self.preprocess_input(image_batch)
        img_batch = np.concatenate(all_img_pre, 0)
        full_prob = sess.run(self.tns_full_prob, feed_dict={self.tns_input: img_batch})
        return full_prob

    def compute_cam(self, image_batch, index_batch):
        """ Compute the CAMs for `image_batch` w.r.t. to the class referenced by `index_batch`
        :param image_batch: the input images for the CAM algorithm
        :param index_batch: the indicies corresponding to the desired class
        :return cam: the CAMs
        """
        activation_batch=[]

        all_img_pre = self.preprocess_input(image_batch)
        if len(all_img_pre) == 1:
            img_batch = all_img_pre[0]
        else:
            img_batch = np.concatenate(all_img_pre, 0)
        
        for input in image_batch:
          input = np.expand_dims(input,0)
          output_conv = self.last_conv_model.predict(input)
          
          # Only first image from convolutions will be used
          resized = resize_activations(output_conv[0], self.input_shape)
          
          # filter_size x input_shape[0] x input_shape[1] - resized to original input dimensions
          normalized_maps = normalize_activations(resized)
          
          # repeat input
          repeat_input = np.tile(self.input, (normalized_maps.shape[0], 1, 1, 1))
        
          expanded_activation_maps = np.expand_dims(normalized_maps, axis=3)
          masked_images = np.multiply(repeat_input, expanded_activation_maps)
          
          # input: filter_size x input_shape[0] x input_shape[1] -> Output filter_size x Classes_Count
          self.classes_activation_scale = self.softmax_model.predict(masked_images,
                                                                    batch_size=self.cam_batch_size)
          
          self.normalized_maps = normalized_maps
          
          #return 3D array
          #return self.normalized_maps

          if self.normalized_maps is None or self.classes_activation_scale is None:
              raise InvalidState('Call prepare_cam before accessing get_class_heatmap, '
                                'activations must be prepared via prepare_cam')
          final_weights = self.classes_activation_scale[:, self.class_id]
          
          final_maps = np.multiply(self.normalized_maps, final_weights.reshape((-1, 1, 1)))
          
          # ReLU
          final_maps_max = np.max(final_maps, axis=0)
          final_class_activation_map = np.where(final_maps_max > 0, final_maps_max, 0)
                  
          #1 Convert 2D array (14,14) into a 2D tensor
          final_class_activation_map = tf.convert_to_tensor(final_class_activation_map, dtype=tf.float32)
          #final_class_activation_map = skimage.transform.resize(final_class_activation_map,)
          
          #2 Add new dimension for batch size
          #final_class_activation_map = tf.expand_dims(final_class_activation_map,0)
          #2 Reshape the 2d (14,14) into (?,14,14) Tensor
          #tns_index = tf.repeat(tns_index , (1, 196) )
          #print(tns_index.shape)

          #3 Cast the tensor type to float32
          final_class_activation_map = tf.dtypes.cast(final_class_activation_map, tf.float32)

          final_class_activation_map = tf.reshape(final_class_activation_map, 
                                                  shape=[-1, tf.shape(final_class_activation_map)[0], 
                                                        tf.shape(final_class_activation_map)[1]])

          activation_batch.append(final_class_activation_map)

        
        self.final_output = tf.concat(activation_batch , 0)

        cam = sess.run(self.final_output, feed_dict={self.tns_input: img_batch, self.tns_index: index_batch})
        print(cam.shape)
        return cam
    
class SingleScoreCamModel:
    def __init__(self, model_input, last_conv_output, softmax_output, input_shape, input, class_id, cam_batch_size=None):
        """
        Prepares class activation mappings
        :param model_input: input layer of CNN, normally takes batch of images as an input. Currently batch must be limited to a single image
        :param last_conv_output: last convolutional layer. The last conv layer contains the most complete information about image.
        :param softmax_output: flat softmax (or similar) layer describing the class certainty
        :param input_shape: Expecting a batch of a single input sample 1 x M X N X ...; it is assumed that 2D image of M x N dimensions is served as an input, which can be multiplied with a 2D-mask.
        :param cam_batch_size: Optional, defaults to None, which will result in inference of batches of size 32.
        """
        self.model_input = model_input
        self.last_conv_output = last_conv_output
        self.softmax_output = softmax_output
        self.last_conv_model = Model(inputs=model_input, outputs=last_conv_output)
        self.softmax_model = Model(inputs=model_input, outputs=softmax_output)
        self.input_shape = input_shape
        self.cam_batch_size = cam_batch_size
        self.input = input
        self.class_id = class_id
        self.normalized_maps = None
        self.classes_activation_scale = None
        
        output_conv = self.last_conv_model.predict(input)
        
        # Only first image from convolutions will be used
        resized = resize_activations(output_conv[0], self.input_shape)
        
        # filter_size x input_shape[0] x input_shape[1] - resized to original input dimensions
        normalized_maps = normalize_activations(resized)
        
        # repeat input
        repeat_input = np.tile(input, (normalized_maps.shape[0], 1, 1, 1))
       
        expanded_activation_maps = np.expand_dims(normalized_maps, axis=3)
        masked_images = np.multiply(repeat_input, expanded_activation_maps)
        
        # input: filter_size x input_shape[0] x input_shape[1] -> Output filter_size x Classes_Count
        self.classes_activation_scale = self.softmax_model.predict(masked_images,
                                                                   batch_size=self.cam_batch_size)
        
        self.normalized_maps = normalized_maps

        if self.normalized_maps is None or self.classes_activation_scale is None:
            raise InvalidState('Call prepare_cam before accessing get_class_heatmap, '
                               'activations must be prepared via prepare_cam')
        final_weights = self.classes_activation_scale[:, class_id]
        
        final_maps = np.multiply(self.normalized_maps, final_weights.reshape((-1, 1, 1)))
        
        final_maps_max = np.max(final_maps, axis=0)
        final_class_activation_map = np.where(final_maps_max > 0, final_maps_max, 0)

        self.final_class_activation_map = final_class_activation_map

class Augmenter:

    def __init__(self, num_aug, augcamsz=(224, 224)):
        self.num_aug = num_aug
        self.augcamsz = augcamsz
        self.tns_angle = tf.placeholder("float", [num_aug])
        self.tns_shift = tf.placeholder("float", [num_aug, 2])
        self.tns_input_img = tf.placeholder("float", [1, augcamsz[0], augcamsz[1], 3])
        self.tns_img_batch = tf.placeholder("float", [num_aug, augcamsz[0], augcamsz[1], 3])

        # tensors for direct augmentation (input transformation)
        tns_img_exp = tf.tile(tf.expand_dims(self.tns_input_img[0], 0), [num_aug, 1, 1, 1])
        tns_rot_img = tf.contrib.image.rotate(tns_img_exp, self.tns_angle, interpolation="BILINEAR")
        self.tns_input_aug = tf.contrib.image.translate(tns_rot_img, self.tns_shift, interpolation="BILINEAR")

        # tensors for inverse augmentation (input anti-transformation)
        tns_shift_img_batch = tf.contrib.image.translate(self.tns_img_batch, self.tns_shift, interpolation="BILINEAR")
        self.tns_inverse_img_batch = tf.contrib.image.rotate(tns_shift_img_batch, self.tns_angle, interpolation="BILINEAR")

    def direct_augment(self, img, angles, shift):
        """ Apply rotation and shift to `img` according to the values in `angles` and `shift`.
        :param img: the input image
        :param angles: the magnitude of the rotation in radians
        :param shift: the magnitude of the shift
        :return img_aug: the transformed image
        """
        feed_dict = {self.tns_input_img: img[np.newaxis, :, :, :],
                     self.tns_angle: angles,
                     self.tns_shift: shift,
                     }
        img_aug = sess.run(self.tns_input_aug, feed_dict)

        return img_aug

    def inverse_augment(self, img_batch, angles, shift):
        """ Apply the inverse rotation and shift to `img_batch` according to the values in `angles` and `shift`.
        :param img_batch: a set of images to be anti-transformed
        :param angles: the magnitude of the rotatation in radians
        :param shift: the magnitude of the shift
        :return img_aug: the anti-transformed image
        """
        feed_dict = {self.tns_img_batch: img_batch,
                     self.tns_angle: -np.array(angles),
                     self.tns_shift: -np.array(shift),
                     }
        img_aug = sess.run(self.tns_input_aug, feed_dict)

        return img_aug


class Superresolution:

    def __init__(self, augmenter, learning_rate=0.001, camsz=(224, 224)):
        num_aug = augmenter.num_aug
        self.augmenter = augmenter
        augcamsz = self.augmenter.augcamsz

        # placeholder tensor for the batch of CAMs resulting from augmentation
        self.tns_cam_aug= tf.placeholder("float", [None, camsz[0], camsz[1], 1])

        # placeholder tensors for the regularization coefficients
        self.tns_lmbda_eng = tf.placeholder("float", [1], name="lambda_eng")
        self.tns_lmbda_tv = tf.placeholder("float", [1], name="lambda_tv")
        # variable tensor for the target upsampled CAM
        self.tns_cam_full = tf.Variable(tf.zeros([1, augcamsz[0], augcamsz[1], 1]), name="cam_full")
        # augmentation parameters tensors
        tns_rot_cam = tf.contrib.image.rotate(tf.tile(self.tns_cam_full, [num_aug, 1, 1, 1]),
                                              augmenter.tns_angle,
                                              interpolation="BILINEAR")
        tns_aug = tf.contrib.image.translate(tns_rot_cam, augmenter.tns_shift, interpolation="BILINEAR")
        # tensor for the downsampling operator
        tns_Dv = tf.expand_dims(tf.image.resize(tns_aug, camsz, name="downsampling"), 0)
        # tensor for the gradient term
        tns_gradv = tf.image.image_gradients(self.tns_cam_full)

        # tensors for the functional terms
        tns_df = tf.reduce_sum(tf.squared_difference(tns_Dv, self.tns_cam_aug), name="data_fidelity")
        tns_tv = tf.reduce_sum(tf.add(tf.abs(tns_gradv[0]), tf.abs(tns_gradv[1])))
        tns_norm = tf.reduce_sum(tf.square(self.tns_cam_full))

        # loss definition
        self.tns_functional_tv = tf.add(tns_df, tf.scalar_mul(self.tns_lmbda_tv[0], tns_tv), name="loss_en_grad")
        self.tns_functional_mixed = tf.add(tf.scalar_mul(self.tns_lmbda_eng[0], tns_norm), self.tns_functional_tv)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.minimizer_mixed = self.optimizer.minimize(self.tns_functional_mixed)
        self.tns_super_single = tf.image.resize(self.tns_cam_aug[0], augcamsz) # first CAM is the non-augmented one

        tns_img_batch = tf.image.resize(self.tns_cam_aug, augcamsz)
        tns_shift_img_batch = tf.contrib.image.translate(tns_img_batch,
                                                         self.augmenter.tns_shift,
                                                         interpolation="BILINEAR")
        self.tns_inverse_img_batch = tf.contrib.image.rotate(tns_shift_img_batch,
                                                             self.augmenter.tns_angle,
                                                             interpolation="BILINEAR")
        self.tns_max_aggr = tf.reduce_max(self.tns_inverse_img_batch, axis=0)
        self.tns_avg_aggr = tf.reduce_mean(self.tns_inverse_img_batch, axis=0)

        all_initializers = []
        all_initializers.append(self.tns_cam_full.initializer)
        for var in self.optimizer.variables():
            all_initializers.append(var.initializer)

        self.all_initializers = all_initializers

    def super_mixed(self, cams, angles, shift, lmbda_tv, lmbda_eng, niter=200):
        """ Compute the CAM solving the super resolution problem.
        :param cams: batch of CAMs
        :param angles: rotation values used to compute each CAM in `cams`
        :param shift: shift values used to compute each CAM in `cams`
        :param lmbda_tv: coefficient promoting the total variation
        :param lmbda_eng: coefficient promoting the energy of the gradient
        :param niter: number of iterations of the gradient descent
        :return cam_full_aug: the upsampled CAM resluting from super resolution aggregation
        """
        feed_dict = {self.tns_cam_aug: cams[:, :, :],
                     self.augmenter.tns_angle: angles,
                     self.augmenter.tns_shift: shift,
                     self.tns_lmbda_tv: [lmbda_tv],
                     self.tns_lmbda_eng: [lmbda_eng]
                     }

        sess.run(self.all_initializers)
        for ii in range(niter):
            _, func = sess.run([self.minimizer_mixed, self.tns_functional_mixed], feed_dict=feed_dict)
            print("{0:3d}/{1:3d} -- loss = {2:.5f}".format(ii+1, niter, func))
        cam_full_aug = sess.run(self.tns_cam_full)

        return cam_full_aug

    def overlay(self, cam, img, th=27, name=None):
        """ Overlay `cam` to `img`.
        The overlay of `cam` to `img` is done cutting away regions in `cam` below `th` for better visualization.
        :param cam: the class activation map
        :param img: the test image
        :param th: threshold for regions cut away from `cam`
        :param name: optional output filename
        :return o: the heatmap superimposed to `img`
        """
        # rotate color channels according to cv2
        background = cv2.cvtColor(img.squeeze(), cv2.COLOR_RGB2BGR)
        foreground = np.uint8(cam / cam.max() * 255)
        foreground = cv2.applyColorMap(foreground, cv2.COLORMAP_JET)

        # mask the heatmap to remove near null regions
        gray = cv2.cvtColor(foreground, code=cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
        mask = cv2.merge([mask, mask, mask])
        masked_foreground = cv2.bitwise_and(foreground, mask)

        BLUE_MIN = np.array([75, 50, 50],np.uint8)
        BLUE_MAX = np.array([130, 255, 255],np.uint8)
        hsv_img = cv2.cvtColor(foreground,cv2.COLOR_BGR2HSV)
        frame_threshed = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
        frame_threshed = cv2.bitwise_and(foreground, foreground, mask=frame_threshed)
        frame_threshed = np.array(frame_threshed , dtype=np.uint8)
        subtract_im = cv2.subtract(foreground, frame_threshed)
        Z = cv2.addWeighted(background, 0.6, subtract_im, 0.4, 0)
        cv2.imwrite(name + "-FILTER.png", Z)
        #********************** Overlay image and blackout background ******************
        blank_image = np.zeros((224,224,3), np.uint8)
        Y1 = cv2.addWeighted(blank_image, 0.5, subtract_im, 0.5, 0)
        edges = cv2.Canny(Y1,60,200)
        #cv2.imwrite("ASC edges-before.png", edges)
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print("Number of ASC Contours is: " + str(len(cnts)))
        for c in cnts:
            cv2.drawContours(edges,[c], -1, (255,255,255), -1)
        #cv2.imwrite("ASC edges-after.png", edges)
        result = cv2.bitwise_and(background,background, mask= edges)
        cv2.imwrite(name + "-BLACKOUT.png", result)
        
        masked_foreground = np.array(masked_foreground , dtype=int)
        background = np.array(background , dtype=int)
        #cv2.imwrite("background" + ".png", background)
        #cv2.imwrite("foreground" + ".png", masked_foreground)


        # overlay heatmap to input image
        o = cv2.addWeighted(background, 0.6, masked_foreground, 0.4, 0)
        if name != None:
            #cv2.imwrite("./imgs/" + name + ".png", o)
            cv2.imwrite(name + ".png", o)
            #plt.savefig(name + ".png")

        return cv2.cvtColor(subtract_im, code=cv2.COLOR_BGR2GRAY)
