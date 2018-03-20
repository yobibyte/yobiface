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

def preproc(im):
  im = cv2.resize(im, (160,160))
  return im.astype(np.float32)/255

def get_random_color():
    return [np.random.random() for i in range(3)]
