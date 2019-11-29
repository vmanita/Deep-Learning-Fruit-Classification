#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:46:28 2019

@author: Manita
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def make_metrics(model, x, y,  class_labels_dict,path=None, name='', save =False):
    probs = model.predict(x)
    predictions = probs.argmax(axis=-1)

    y_true = [list(i).index(1) for i in y]
    
    if save:
        predict_df = pd.DataFrame({'y_real':y_true, 'y_predict':predictions})
        predict_df.to_excel(path+name+".xlsx")
 
    class_report = classification_report(y_true, predictions, target_names=class_labels_dict.values())
    
    print(class_report)



def plot_accuracy(history):
        
    keys_ = list(history.history.keys())  
    acc = history.history[keys_[1]]
    val_acc = history.history[keys_[3]]
    plt.figure(figsize = (10,5))
    # train
    plt.plot(acc, '-',label='Training ' + keys_[1])
    # valid
    plt.plot(val_acc, '--', label='Validation ' + keys_[1])
    plt.title(keys_[1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
def plot_loss(history):
        
    keys_ = list(history.history.keys())  
    loss = history.history[keys_[0]]
    val_loss = history.history[keys_[2]]
    plt.figure(figsize = (10,5))
    # train
    plt.plot(loss, '-',label='Training loss')
    # valid
    plt.plot(val_loss, '--', label='Validation loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
from keras import backend as K




def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def plot_image_sample(x, y, labels):
    
    fig = plt.figure(figsize =(10,6))
    for i in range(9):
        index_y = list(y[i]).index(1)
        ax = fig.add_subplot(3,3,i+1,xticks=[],yticks=[])
        ax.title.set_text('actual class: {}'.format(labels[index_y]))
        ax.imshow(np.squeeze(x[i]))
    

def compare_models_output(models_dict, x_to_compare, y_to_compare):
    
    y_true = [list(i).index(1) for i in y_to_compare]

    compare_models = pd.DataFrame({'y_true': y_true})
    
    for key, model in tqdm(models_dict.items()):
        predict_prob = model.predict(x_test)
        predictions = predict_prob.argmax(axis=-1)
        
        compare_models[key] = predictions
        
    return compare_models


def compare_models(models_dict, x_to_compare, y_to_compare):
    compare_mod = compare_models_output(models_dict, x_to_compare, y_to_compare)
    
    metrics_df = pd.DataFrame(index = models_dict.keys())

    f1_concat = []
    acc_concat = []
    precision_concat = []
    recall_concat = []

    for key, model in models_dict.items():
        
        acc_ = accuracy_score(compare_mod['y_true'], compare_mod[key])
        f1_ = f1_score(compare_mod['y_true'], compare_mod[key], average='weighted')
        prec_ = precision_score(compare_mod['y_true'], compare_mod[key],  average='weighted')
        rec_ = recall_score(compare_mod['y_true'], compare_mod[key],  average='weighted')
        
        f1_concat.append(f1_)
        acc_concat.append(acc_)
        precision_concat.append(prec_)
        recall_concat.append(rec_)
        
    metrics_df['acc'] = acc_concat
    metrics_df['f1'] = f1_concat
    metrics_df['prec'] = precision_concat
    metrics_df['rec'] = recall_concat

    return metrics_df



