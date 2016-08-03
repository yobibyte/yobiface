# yobiface
Usage of https://github.com/davidsandberg/facenet/ pre-trained model for faces identification

## Dependencies

* tensorflow
* opencv with python bindings (cv2)
* numpy
* sklearn for t-SNE and olivetti dataset
* matplotlib
* jupyter notebook for running .ipynb examples

## How to use

* go to models folder and write the full paths to checkpoint file in the same folder
* put [lfw](http://vis-www.cs.umass.edu/lfw/lfw.tgz) dataset to data folder
* open lfw.ipynb or olivetti.ipynb for examples

## TODO

* add confidence of prediction
* add flask server for incoming requests
* store embeddings in a file and load when start a server
* add data loader for lfw
* fix the problem due to which I need to reload the whole notebook to rerun cell with model loading
