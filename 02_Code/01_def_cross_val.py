#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras import models, Sequential, layers
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
import os, shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import multiprocessing as mp
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tqdm import tqdm
from keras.utils import np_utils
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.datasets import load_files
import numpy as np
import seaborn as sns

#******************************************************************************
# Import ---> CHANGE PATHS HERE
#******************************************************************************


#base_dir = '/Users/Manita/Documents/dl_data/Fruits'
#path= '/Users/Manita/OneDrive - NOVAIMS/Deep Learning/Project'
#check_path = '/Users/Manita/OneDrive - NOVAIMS/Deep Learning/Project/'

#path='D:\\NOVA IMS\\Deep Learning\\'
#check_path='C:\\Users\\vitor\\OneDrive - NOVAIMS\\Deep Learning\\Project\\models_backup\\'
#log_path = 'C:\\Users\\vitor\\OneDrive - NOVAIMS\\Deep Learning\\Project\\log\\'
#base_dir = 'D:\\NOVA IMS\\Deep Learning\\Fruits\\'
check_path = '/Users/Manita/OneDrive - NOVAIMS/Deep Learning/Project/models_backup/'
log_path = '/Users/Manita/OneDrive - NOVAIMS/Deep Learning/Project/log/'
base_dir = '/Users/Manita/Documents/dl_data/Fruits/'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

initial_gen = ImageDataGenerator(rescale=1./255,                              
                                 validation_split=0.1)
test_gen = ImageDataGenerator(rescale=1./255)


# Parameters *************
n_folds = 5
n_batches = 32
n_epochs = 25
n_classes = 16
image_size = 50
early_stopping_criteria = 'val_loss'
n_epochs_before_stopping = 10
# *************************   

# Plot metrics
def box_plot(df_metrics):

    plt.figure(figsize = (15,5))
    ax = sns.boxplot(data = df_metrics, 
                     boxprops=dict(alpha=.8),linewidth=1)
    
    sns.stripplot(data = df_metrics,
                  size=3, jitter=True, edgecolor="black", linewidth=.5)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize =12) 
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    plt.xlabel('Metric',  fontweight = "bold")
    plt.ylabel('Metric Score',  fontweight = "bold")
    plt.show()
    
    
    
def cross_val_model(model, name, img_size, boxplot=False, check_path=check_path, log_path = log_path):
    
    f1_scores = []
    acc_scores = []
    precision_scores = []
    recall_scores = []
    
    for seed in range(n_folds):
        
        print('\n>>> Starting Fold Nr.: {}'.format(seed+1))
 
        # Import all training images and split into train and valid
        init_train_gen = initial_gen.flow_from_directory(
                train_dir,
                target_size=(img_size, img_size),
                batch_size=10000,
                class_mode='categorical',
                subset='training',
                shuffle = False,
                seed = seed)
        
        init_valid_gen = initial_gen.flow_from_directory(
                train_dir,
                target_size=(img_size, img_size),
                batch_size=10000,
                class_mode='categorical',
                subset ='validation',
                shuffle = False,
                seed = seed)
        
        x_train, y_train = init_train_gen.next()
        x_val, y_val = init_valid_gen.next()
        print('\nTrain size: {}'.format(len(x_train)))
        print('Validation size: {}'.format(len(x_val)))
        
        # start generator for transformations
        
        # training
        train_gen = ImageDataGenerator(horizontal_flip = False,
                                       vertical_flip = False,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       zoom_range = 0.1)
        

        train_generator = train_gen.flow(x_train, y_train, batch_size = n_batches)
        #valid_generator = test_gen.flow(x_val, y_val, batch_size = 10)
        
        # Save best model for each seed
        checkpointer = ModelCheckpoint(filepath = check_path+name+'_'+str(seed)+'.hdf5',
                                       verbose = 1, save_best_only = True)
        
        # Early stop callback
        early_stop = EarlyStopping(monitor = early_stopping_criteria,
                                   patience = n_epochs_before_stopping, verbose = 1)

        img_shape = (img_size, img_size, 3)
        
        history = model.fit_generator(
                train_generator,
                steps_per_epoch= len(train_generator),
                epochs=n_epochs,
                shuffle=True,
                verbose=1,
                validation_data = (x_val, y_val),
                callbacks = [checkpointer, early_stop])

        plot_accuracy(history)
        plot_loss(history)
        
        #exporting history     filepath = check_path+name+str(seed)+'.hdf5'
        history_df = pd.DataFrame()
        history_df['loss'] = history.history['loss']
        history_df['val_loss'] = history.history['val_loss']
        history_df['f1'] = history.history['f1']
        history_df['val_f1'] = history.history['val_f1']
        
        history_df.to_excel(log_path+'hist_'+name+'_'+str(seed)+'.xlsx')
        
        # predict on entire validation
        
        probs = model.predict(x_val)
        predictions = probs.argmax(axis=-1)
        
        y_true = [list(i).index(1) for i in y_val]
        
        f1_ = f1_score(y_true, predictions, average='weighted')
        acc_ = accuracy_score(y_true, predictions)
        prec_ = precision_score(y_true, predictions,  average='weighted')
        rec_ = recall_score(y_true, predictions,  average='weighted')
        
        print(' ACCURACY: {}'.format(np.round(acc_, 2)))
        print(' F1 SCORE: {}'.format(np.round(f1_, 2)))
        print('\n')
        
        f1_scores.append(f1_)
        acc_scores.append(acc_)
        precision_scores.append(prec_)
        recall_scores.append(rec_)

    # Create a dictionary with metrics of each fold and export
    models_dict = {}
    metrics = {'f1':f1_scores,
               'acc':acc_scores,
               'pre':precision_scores,
               'rec':recall_scores}

    for key_metric, metric_list in metrics.items():
        models_dict[name+'_'+ key_metric] = metric_list
            
    df = pd.DataFrame.from_dict(metrics, orient='columns')
    df.to_excel(log_path+name+'.xlsx')

    if boxplot:
        box_plot(df)



    
    
 

 




  
























