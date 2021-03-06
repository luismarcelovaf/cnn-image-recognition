# -------------------------------------------------------
# -------------------- MyCoco ---------------------------
# - Solution for Coco classification problem		-
# - using convolutional neural networks applied in	-
# - theano using keras superlayer			-
# -------------------------------------------------------
# ---------	Author : Luis Marcelo Fonseca	---------
# ---------	Last Edit : 15/11/2015		---------
# -------------------------------------------------------

# ---------- Imports ---------
# Keras API
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# COCO API
from pycocotools.coco import COCO
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pylab

# Python API
import numpy as np
import math
import sys
# ----------------------------

# ------ Configurations ------

# **** Path settings *****

# Data directory (coco root folder)
dataDir='..';

# ** Trainning settings **

# Image size in pixels fed to the CNN
resizeX = 32;
resizeY = 32;

batch_size = 32;
nb_classes = 91;
nb_epoch = 5;

# Optimizer to be used in model
# sgd, adadelta, adagrad, rmsprop, ...
modelOptimizer = 'adagrad';

# If using SGD, specify learning rate, weight decay, momentum.
SgdLR = 0.1;
SgdDecay = 1e-6;
SgdMomentum = 0.9;
SgdNesterov = True;

# If set to true, will use only data from validation and split into train/val
datasetSplit = True;
splitPercentage = 0.7;

# ------ Coco Settings -------

# Categories to be trained (remove to train in all categories)
# Will load all images with either of the categories or all of them (union of sets)
catRestriction = True;
catNms=['airplane', 'car', 'bird', 'cat', 'sheep' , 'dog', 'giraffe', 'horse', 'bicycle', 'truck' ];

# Missing images from dataset existing on annotations
imgIdBlackList = [167126];

# Lists with image IDs of instaces processed
imgIdLst = [];

# ----------------------------

# --- Auxiliary Functions ----

# Receives one A x B shape matrix and converts it into
# three A x B matrices of same values. (copying values)
def convert_bw_rgb(img):

	shpX, shpY = img.shape;
	ret = np.ndarray((shpX, shpY, 3));
	for i in range(0, shpX):
		for j in range(0, shpY):
			for k in range(0, 3):
				ret[i,j,k] = img[i,j];

	return ret;

# Switches shape from (X, Y, Channel) in whatever size
# To (Channel, X, Y) in 32x32 pixels
def convert_model_shape(img):

	# Skimage resize function
	img = resize(img,(resizeX, resizeY));
	imt = img.astype("float32");
	img = img.reshape(3, resizeX, resizeY);
	return img;

# Shuffles two arrays together
def shuffle_in_unison(a, b):
	assert len(a) == len(b)
	shuffled_a = np.empty(a.shape, dtype=a.dtype);
	shuffled_b = np.empty(b.shape, dtype=b.dtype);
	permutation = np.random.permutation(len(a));
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index];
		shuffled_b[new_index] = b[old_index];
	return shuffled_a, shuffled_b;

# Function returns that returns n-dim vectors with loaded data
# valSet must be true if received data should be loaded as validation set
# valSet should be left as default with dataset split enabled
def load_data(setDir, valSet=False):
	
	# initialize COCO api for trainning annotations
	coco=COCO('%s/annotations/instances_%s.json'%(dataDir,setDir));

	# Gets all images containing given categories (union of sets)
	if (catRestriction == True):
		catIds = coco.getCatIds(catNms);
	else:
		catIds = coco.getCatIds();

	imgIds = []
	for i in range(len(catIds)):
		imgIds += coco.getImgIds(catIds=catIds[i]);

	# Loads the image objects list
	imgObjList = coco.loadImgs(imgIds);

	print len(imgObjList),' images loaded';

	# X is the ndarray of features
	# Y is the ndarray of classes
	X = [];
	Y = [];

	print 'initializing image processing (',resizeX,'x',resizeY,' pixels resize)...';

	# X receives each img RGB pixel matrix
	count = 0;
	for i in range(len(imgObjList)):

		# Verifies if image is not in excluded images list
		if (imgObjList[i]['id'] not in imgIdBlackList):

			# Data taken from imread is already normalized
			img = io.imread('%s/images/%s/%s'%(dataDir,setDir,imgObjList[i]['file_name']));
		
			# Convert B&W image into RGB
			# This will copy all values from B&W width x height pixel matrix into
			# All three R, G, B, width x height pixel matrices.
			if (img.ndim <> 3):
				img = convert_bw_rgb(img);

			# Obtains annotation for the given image
			AnnIds = coco.getAnnIds(imgIds=imgObjList[i]['id'])
			AnnObjList = coco.loadAnns(ids=AnnIds);

			# Processes each instance (ocurrence of each object in an image)
			for j in range(len(AnnObjList)):

				bbX, bbY, bbW, bbH = AnnObjList[j]['bbox'];

				# Checks for broken bounding boxes
				if ((int(bbW) > 0) and (int(bbH) > 0)):
					# Checks if the annotation belongs to the selected categories
					if ((catRestriction == False) or (AnnObjList[j]['category_id'] in catIds)):
						# Crop from x, y, w, h -> 100, 200, 300, 400
						# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
						crop = img[bbY:bbH+bbY,bbX:bbW+bbX].copy();

						# Fill category vector
						Y.append(AnnObjList[j]['category_id']);

						# Place ID of processed instance into list
						if (valSet == True):
							imgIdLst.append(imgObjList[i]['id']);

						# Fill input matrix
						inst = convert_model_shape(crop);
						X.append(inst);

						# Counter of how many images have been successfully processed
						count += 1;

		if (((i % (math.floor(len(imgObjList)/20))) == 0) and (i != 0)):
			print 'Images processed: ', i;

	# i starts at zero, therefore we add one to count the number of iterations we made
	print 'Instances processed: ', count;
	print '\n';

	X = np.asarray(X);
	Y = np.asarray(Y);

	# Shuffles the arrays together
	# Disabled because imgIdLst isn't shuffled as well
	X, Y = shuffle_in_unison(X, Y);

	return X, Y;

# First CNN solution based on cifar-10 and adapted
def build_model_cnn1():

	model = Sequential();

	model.add(Convolution2D(32, 3, 3, border_mode='full',
	input_shape=(3, resizeX, resizeY)));
	model.add(Activation('relu'));
	model.add(Convolution2D(64, 3, 3));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.2));

	model.add(Convolution2D(32, 3, 3, border_mode='full'));
	model.add(Activation('relu'));
	model.add(Convolution2D(64, 3, 3));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.2));

	model.add(Flatten());
	model.add(Dense(2048));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.2));
	model.add(Dense(1024));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.2));
	model.add(Dense(512));
	model.add(Activation('relu'));
	model.add(Dropout(0.2));

	model.add(Dense(nb_classes));
	model.add(Activation('softmax'));

	'''
	model.add(Convolution2D(32, 3, 3, border_mode='full',
	input_shape=(3, resizeX, resizeY)));
	model.add(Activation('relu'));
	model.add(Convolution2D(32, 3, 3));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.25));

	model.add(Convolution2D(64, 3, 3, border_mode='full'));
	model.add(Activation('relu'));
	model.add(Convolution2D(64, 3, 3));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Dropout(0.25));

	model.add(Flatten());
	model.add(Dense(512));
	model.add(Activation('relu'));
	model.add(Dropout(0.5));
	model.add(Dense(nb_classes));
	model.add(Activation('softmax'));
	'''

	# let's train the model using SGD + momentum (how original).
	if (modelOptimizer == 'sgd'):
		sgd = SGD(lr=SgdLR, decay=SgdDecay, momentum=SgdMomentum, nesterov=SgdNesterov)
		model.compile(loss='categorical_crossentropy', optimizer=sgd);
		print 'Using Stochastic Gradient Descent.';
		print 'Learning Rate:',SgdLR;
		print 'Decay:',SgdDecay;
		print 'Momentum:',SgdMomentum;
		print 'Nesterov:',SgdNesterov;
	else:		
		model.compile(loss='categorical_crossentropy', optimizer=modelOptimizer);

	return model;

# ----------------------------

if __name__ == "__main__":
	
	# ** Data Processing **
	print '\nLoading data...';

	# Loads train data into memory
	if (datasetSplit == True):
		print 'Loading Train/Val data...';
		X, Y = load_data('train2014');

		# Splits into train/val
		print 'Splitting Train/Val data (',splitPercentage * 100,'% Split)...';
		splitInd = int(splitPercentage*len(X));
		X_train = []; Y_train = []; X_test = []; Y_test = [];
		for i in range(splitInd):
			X_train.append(X[i]);
			Y_train.append(Y[i]);
		for i in range(splitInd, len(X)):
			X_test.append(X[i]);
			Y_test.append(Y[i]);
		X_train = np.asarray(X_train); X_test = np.asarray(X_test);
		Y_train = np.asarray(Y_train); Y_test = np.asarray(Y_test);
	else:
		# Loads each set independently
		print 'Loading Train data...';
		X_train, Y_train = load_data('train2014', valSet=False);
		print 'Loading Validation data...';
		X_test, Y_test = load_data('val2014', valSet=True);

	# Print shapes
	print 'X_train shape: ', X_train.shape;
	print 'Y_train shape: ', Y_train.shape;
	print 'X_test shape: ', X_test.shape;
	print 'Y_test shape: ', Y_test.shape;

	# Convert class vector to binary class matrix, for use with categorical_crossentropy
	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	Y_test = np_utils.to_categorical(Y_test, nb_classes)
	
	# ** Model Building **
	print '\nBuilding model...';
	
	# Calls function that builds CNN
	model = build_model_cnn1();

	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True);
	
	Y_pred = model.predict(X_test, batch_size=30);

	# Converts Y_pred and Y_test into vector to use in confusion matrix
	# Also recovers numbers of occurences of instances for each category in val set
	catInstNum = np.zeros((nb_classes));
	vetLab = [];
	for i in range(0,nb_classes):
		vetLab.append(i);
	Y_vetPred = [];
	Y_vetTest = [];
	if (len(Y_pred) == len(Y_test)):
		for i in range(0,len(Y_pred)):
			predMax = 0;
			indMax = -1;
			for j in range(0,nb_classes):
				if (Y_pred[i, j] > predMax):
					predMax = Y_pred[i, j];
					indMax = j;
				if (Y_test[i, j] == 1):
					Y_vetTest.append(j);
					catInstNum[j] += 1;
			Y_vetPred.append(indMax);
	else:
		print 'Error: Cannot print confusion matrix, Y_pred and Y_test have different length';
	
	# Prints all COCO Categories and their IDs
	coco=COCO('%s/annotations/instances_val2014.json'%(dataDir));
	catObjs = coco.loadCats(coco.getCatIds());
	strCats = "";
	idCont = 1;
	print '\nID/Category/SuperCategory';
	for i in range(0,len(catObjs)):
		while (catObjs[i]['id'] <> idCont):
			strCats += str(idCont)+'/?/?,';
			idCont += 1;

		strCats += str(catObjs[i]['id']) +'/'+ catObjs[i]['name'] + '/'+catObjs[i]['supercategory']+',';
		idCont += 1;
	strCats += '91/unknown/unknown';
	print strCats;

	# Prints number of instances per category
	print '\nNumber of instaces per category (val set only)';
	strNum = "";
	for i in range(0,len(catInstNum)):
		strNum += str(catInstNum[i]) + ',' ;
	print strNum;

	# Print Options
	np.set_printoptions(threshold='nan')  # For printing whole matrix
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	# Prints confusion matrix
	print '\nConfusion matrix:';
	print confusion_matrix(Y_vetTest, Y_vetPred);
	print '\n'

	# Used to print IDs of images that were cats but predicted as cars (for example)
	# imgPerCat receives the number of different image IDs to be displayed
	# Only displays when not using dataset split.
	'''
	if (datasetSplit == False):
		print '\nSample mistaken predictions:';
		cmpCount1 = 0;
		cmpCount2 = 0;
		cmpMaxCount = 7; # Displays 3 occurrences
		cmpPrinted = [];
		idCatPred = 3; # Predicted: Car
		idCatReal = 8; # Real: Truck
		for i in range(0,len(Y_vetPred)):
			if ((cmpCount1 >= cmpMaxCount) and (cmpCount2 >= cmpMaxCount)):
				break;
			if (imgIdLst[i] not in cmpPrinted):
				if ((Y_vetPred[i] == idCatPred) and (Y_vetTest[i] == idCatReal)):
					if (cmpCount1 < cmpMaxCount):
						print 'Predicted category ID:',Y_vetPred[i],'- Real Category ID:',Y_vetTest[i],'- Image ID:', imgIdLst[i];
						cmpPrinted.append(imgIdLst[i]);
						cmpCount1 += 1;
				elif ((Y_vetPred[i] == idCatReal) and (Y_vetTest[i] == idCatReal)):
					if (cmpCount2 < cmpMaxCount):
						print 'Predicted category ID:',Y_vetPred[i],'- Real Category ID:',Y_vetTest[i],'- Image ID:', imgIdLst[i];
						cmpPrinted.append(imgIdLst[i]);
						cmpCount2 += 1;
	'''
	# Prints classification report with metrics
	print '\nClassification Report:';
	orderedCatNms = [];
	for i in range(0,len(catObjs)):
		if (catObjs[i]['name'] in catNms):
			orderedCatNms.append(catObjs[i]['name']);
	if (catRestriction == True):
		print(classification_report(Y_vetTest, Y_vetPred, target_names=orderedCatNms))

	# Evaluates accuracy on val set
	print 'Evaluating score...';
	score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True);
	print('Test score (Avg. Recall):', score);

