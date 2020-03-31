# author: "Yixing Wang, Northwestern University"
# copyright: = "Advanced Materials Lab"
# email = "yixingwang2014@u.northwestern.edu"

import keras
import os
import scipy.io
import pandas as pd
import numpy as np
from numpy import zeros, newaxis
from keras import models
from keras.layers import core, convolutional, pooling,Dropout
from keras import models, optimizers, backend,Input
from keras.models import Model
from sklearn.cross_validation import train_test_split
import pickle
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

def read_input(data_file,interphase_image,ratio,number):
	X,Y = load_data(data_file,interphase_image)

	Y = np.transpose(Y)
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size = ratio, random_state = number)
	X_val, X_test, Y_val, Y_test = train_test_split(
		X_test, Y_test, test_size = 0.5, random_state = number)
	r_train = Y_train[:,0]
	r_test = Y_test[:,0]
	g_train = Y_train[:,1]
	g_test = Y_test[:,1]
	t_train = Y_train[:,2]
	t_test = Y_test[:,2]
	return X_train,X_test,[r_train,g_train,t_train],[r_test,g_test,t_test]

def load_data(data_file,interphase_image):
	data = pd.read_csv(data_file)
	data = data.sample(frac = 0.2,random_state = 1)
	img_files = list(data.img_file)
	images = []

	for i in range(len(img_files)):
		if not interphase_image:
			image = scipy.io.loadmat(img_files[i])
			image = image['img_out'] - 0.5
			image = image[:,:,newaxis]
			images.append(image)
		else:
			np_image = interphase_image + '/' + img_files[i].split('/')[2] + '.npy'
			images.append(np.load(np_image)[:,:,newaxis])
	Y = [list(data.rub_mods),list(data.glass_mods),list(data.tan_delta)]

	return np.asarray(images),np.asarray(Y)



def multi_task_model(loss_func,ratio):
	#shared cnn architecture
	model_input = Input(shape = (256,256,1))
	shared_layer = Conv2D(16, kernel_size = (3,3),activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(model_input)
	shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
	shared_layer = Conv2D(32, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
	shared_layer = Conv2D(64, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
	shared_layer = Conv2D(128, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)

	#multitask training
	#rub_mods
	rub_mods = Conv2D(128, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	rub_mods = MaxPooling2D(pool_size=(2, 2))(rub_mods)
	rub_mods = core.Flatten()(rub_mods)
	rub_mods = core.Dense(512, activation='relu')(rub_mods)
	rub_mods = Dropout(ratio)(rub_mods)
	rub_mods = core.Dense(256, activation='relu')(rub_mods)
	rub_mods = Dropout(ratio)(rub_mods)
	rub_mods = core.Dense(1)(rub_mods)

	#glass_mods
	glass_mods = Conv2D(128, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	glass_mods = MaxPooling2D(pool_size=(2, 2))(glass_mods)
	glass_mods = core.Flatten()(glass_mods)
	glass_mods = core.Dense(512, activation='relu')(glass_mods)
	glass_mods = Dropout(ratio)(glass_mods)
	glass_mods = core.Dense(256, activation='relu')(glass_mods)
	glass_mods = Dropout(ratio)(glass_mods)
	glass_mods = core.Dense(1)(glass_mods)

	#tan_delta
	tan_delta = Conv2D(128, kernel_size = (3,3), activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0005))(shared_layer)
	tan_delta = MaxPooling2D(pool_size=(2, 2))(tan_delta)
	tan_delta = core.Flatten()(tan_delta)
	tan_delta = core.Dense(512, activation='relu')(tan_delta)
	tan_delta = Dropout(ratio)(tan_delta)
	tan_delta = core.Dense(256, activation='relu')(tan_delta)
	tan_delta = Dropout(ratio)(tan_delta)
	tan_delta = core.Dense(1)(tan_delta)


	model = Model(inputs = model_input,outputs = [rub_mods,glass_mods,tan_delta])
	model.compile(optimizer=optimizers.Adam(lr=1e-04), loss = loss_func)

	return model

def main(data_file,seq,previous_model,number):
	X_train,X_test,Y_train,Y_test = read_input(data_file,'',0.3,number)
	previous_model = ''
	if previous_model:
		model = load_model(previous_model)
	else:
		model = multi_task_model('mean_absolute_percentage_error',0)
	weight_file = 'save_file.hdf5'
	checkpoint1 = keras.callbacks.ModelCheckpoint(weight_file , monitor='val_loss', verbose=1, save_best_only = True, save_weights_only = False, mode='auto', period=1)
	checkpoint2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience = 50 , verbose=0, mode='auto')

	call_back_list = [checkpoint1,checkpoint2]
	history = model.fit(X_train, Y_train,
			  validation_data = (X_test,Y_test),
			  batch_size= 32,
			  epochs = 1000,
			  callbacks = call_back_list)
	hist_file = 'hist_' + str(seq) + '.p'
	with open(hist_file, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)


if __name__ == '__main__':
	for i in range(1):
		main('./training_data.csv',i + 1,'',i + 1)
