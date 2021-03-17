import time
import os
import pathlib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


def convert_to_cv_box(box):
    x = int(box[0]*700)
    y = int(box[1]*700)
    h = int(box[0]*700) + int(box[2]*700)
    w = int(box[1]*700) + int(box[3]*700)
    print(x, y, w, h)
    return (x, y, w, h)


PATH_TO_SAVED_MODEL = "C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/my_model/saved_model"
PATH_TO_LABELS = "C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/training/label_map.pbtxt"
IMAGE_PATHS = [
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha.jpg',
    # 'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha1.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha2.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha3.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha4.jpg',
    # 'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha5.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha6.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha7.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha8.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha9.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha10.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha11.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha12.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha13.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha14.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha15.jpg',
    'C:/Users/Dogukan/Desktop/Miniconda3Proj/Tensorflow/models/research/object_detection/iha16.jpg']

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

print(category_index)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)
    print("**************", image_np)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)
    print("detection classes :", detections['detection_classes'][0])
    print("detection boxes :", detections['detection_boxes'][0])
    print("detection score :", detections['detection_scores'][0])

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.60,
        agnostic_mode=False)

    image_np_with_detections = cv2.cvtColor(
        image_np_with_detections, cv2.COLOR_RGB2BGR)
    x, y, w, h = convert_to_cv_box(detections['detection_boxes'][0])
    # cv2.rectangle(image_np_with_detections,(x,y),(w,h),(255,0,0),2)
    cv2.imshow("detections", image_np_with_detections)
    cv2.waitKey(0)
    print('Done')
