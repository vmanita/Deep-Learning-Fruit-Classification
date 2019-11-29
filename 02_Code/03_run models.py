# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:02:27 2019

@author: PC
"""


#### Rita 
cross_val_model(basic_model(50),'basic_50', 50, True, check_path, log_path)
cross_val_model(basic_model(64),'basic_64', 64, True, check_path)

cross_val_model(lennet_5(50),'lennet_5_50', 50, True, check_path)
cross_val_model(lennet_5(64),'lennet_5_64', 64, True, check_path)

cross_val_model(alexnet(50),'alexnet_50', 50, True, check_path)
cross_val_model(alexnet(64),'alexnet_64', 64, True, check_path)

cross_val_model(cnn_2(50),'cnn_2_50', 50, True, check_path)
cross_val_model(cnn_2(64),'cnn_2_64', 64, True, check_path)



#### Manita  ******************************************************************
# 8 horas a correr
#cross_val_model(vgg16(50),'vgg16_50', 50, True, check_path, log_path)
#cross_val_model(vgg16(64),'vgg16_64', 64, True, check_path, log_path)

cross_val_model(resnet18(50),'resnet18_50', 50, True, check_path, log_path)
cross_val_model(resnet18(64),'resnet18_64', 64, True, check_path, log_path)

cross_val_model(resnet34(50),'resnet34_50', 50, True, check_path, log_path)
cross_val_model(resnet34(64),'resnet34_64', 64, True, check_path, log_path)

cross_val_model(cnn_3(50),'cnn_3_50', 50, True, check_path)
cross_val_model(cnn_3(64),'cnn_3_64', 64, True, check_path, log_path)

#*******************************************************************************

#### Umbelino 
cross_val_model(resnet50(50),'resnet50_50', 50, True, check_path)
cross_val_model(resnet50(64),'resnet50_64', 64, True, check_path)

cross_val_model(resnet101(50),'resnet101_50', 50, True, check_path)
cross_val_model(resnet101(64),'resnet101_64', 64, True, check_path)

cross_val_model(resnet152(50),'resnet152_50', 50, True, check_path)
cross_val_model(resnet152(64),'resnet152_64', 64, True, check_path)

cross_val_model(cnn_sigmoid(50),'cnn_sigmoid_50', 50, True, check_path)
cross_val_model(cnn_sigmoid(64),'cnn_sigmoid_64', 64, True, check_path)



'''
#por implementar
inceptionnet
vgg19
'''



###############################################################################
'''
 TO CHECK
 
 -> BOOST YOU CNN WITH PROGRESSIVE RECIZING
 https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20
 
 -> CENAS DE ACTIVATION FUNCTIONS
 https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
 
 -> DONT USE DROPOUT IN CNN, TRY BATCH NORMALIZATION
 https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16


'''
###############################################################################