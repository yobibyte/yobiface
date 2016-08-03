import numpy as np
import tensorflow as tf
import cv2
import argparse

'''
We load data and convert it to BGR,
in order to show pic, we want to convert it back.
'''
def imshow(fig, pic, rows=1, cols=1, pos=1):
  ax = fig.add_subplot(rows,cols,pos)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.imshow(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB));

def load_tf_flags():
  FLAGS = tf.python.platform.flags._FlagValues()
  tf.app.flags._global_parser = argparse.ArgumentParser()
  tf.app.flags.DEFINE_string('model_dir', '~/dev/facenet/facenet/models',"Directory containing the graph definition and checkpoint files.")
  tf.app.flags.DEFINE_string('model_def', 'models.nn4', "Points to a module containing the definition of the inference graph.")
  tf.app.flags.DEFINE_integer('image_size', 96, "Image size (height, width) in pixels.")
  tf.app.flags.DEFINE_string('pool_type', 'MAX',"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.")
  tf.app.flags.DEFINE_boolean('use_lrn', False, "Enables Local Response Normalization after the first layers of the inception network.")
  tf.app.flags.DEFINE_integer('seed', 42, "Random seed.")
  # None batch size allows to use any batch size
  tf.app.flags.DEFINE_integer('batch_size', None, "Number of images to process in a batch.")
  return FLAGS

def preproc(im):
  im = cv2.resize(im, (96,96))
  # im = np.rollaxis(im, 2, 0)
  return im.astype(np.float32)/255

def get_random_color():
    return [np.random.random() for i in range(3)]
