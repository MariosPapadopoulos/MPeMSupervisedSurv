import os
import argparse
from SurvivalUtils import  TrainAndEvaluateModel, InputFunction
import warnings
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import keras
from pathlib import Path
from sksurv.metrics import concordance_index_censored
import gc
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array 
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
from ast import literal_eval
import numpy as np

def nameFiltering(path):
    
    temp1= os.listdir(path)
    temp1_normal = []
    temp1_aug = []
    
    # Iterate over the list of filenames
    for filename in temp1:
        if 'aug' in filename.lower():
            temp1_aug.append(filename)
        else:
            temp1_normal.append(filename)
            
    return temp1_normal, temp1_aug


if __name__ == '__main__':
    #csv_path: csv file with patient information
    
    file_df=pd.read_csv(csv_path,converters={'TumourTiles': literal_eval,'TLSTiles': literal_eval})
    usable_df= file_df.dropna(subset=['OS time'])
    usable_df = usable_df[usable_df['PatientID'] != '12R01647']  #patient 12R01647 is unusable-too few WSIs
    
    neoadjuvant_chemo_yes=usable_df[usable_df['Adjuvant chemotherapy']=='Yes']
    neoadjuvant_chemo_no=usable_df[usable_df['Adjuvant chemotherapy']=='No']
    
    #run the analysis for adjuvant_chemo_yes:
    df= neoadjuvant_chemo_no
    train_index = df[df['Train_Test']=='Train'].index
    test_index = df[df['Train_Test']=='Test'].index
    df_train=df[df['Train_Test']=='Train'].reset_index(drop=True)
    df_test=df[df['Train_Test']=='Test'].reset_index(drop=True)    
    
    
    num_patches=150
    image_dim= 128
    train_x_tiles = list(df_train['Path'])
    for pos, i in enumerate(train_x_tiles):
        # temp1= os.listdir(i)
        # temp1_normal=
        # # temp1= temp1[0:200]
        temp1_normal, temp1_aug= nameFiltering(i)
        temp1= temp1_normal.copy()
        
        
        if len(temp1_normal)<num_patches:
            temp2= temp1_aug[0: num_patches-len(temp1_normal)]
            temp1.extend(temp2)
        elif len(temp1_normal)>num_patches:
            temp1_normal= temp1_normal[0:num_patches]
            temp1= temp1_normal.copy()
            
        temp3=[i+'/'+name for name in temp1]
        train_x_tiles[pos]= temp3
        
          
    train_time = list(df_train['OS time'])
    train_event = list(df_train['Event'])

    train_x = [load_img(j, target_size=(image_dim, image_dim)) for i in (train_x_tiles) for j in i]  
    train_x = [image.img_to_array(i) for i in train_x]
    train_x = np.asarray(train_x)
    train_x= train_x/255.0
    
    train_time = np.asarray(train_time)
    train_time= np.repeat(train_time, num_patches)
    train_event = np.asarray(train_event)
    train_event= np.repeat(train_event, num_patches)
    
    test_x_tiles= list(df_test['Path'])
    for pos, i in enumerate(test_x_tiles):
    #     temp1= os.listdir(i)
    #     temp1= temp1[0:200]
    #     temp2=[i+'/'+name for name in temp1]
    #     test_x_tiles[pos]= temp2
        temp1_normal, temp1_aug= nameFiltering(i)
        temp1= temp1_normal.copy()
        
        if len(temp1_normal)<num_patches:
            temp2= temp1_aug[0: num_patches-len(temp1_normal)]
            temp1.extend(temp2)
        elif len(temp1_normal)>num_patches:
            temp1_normal= temp1_normal[0:num_patches]
            temp1= temp1_normal.copy()
        
          
        temp3=[i+'/'+name for name in temp1]
        test_x_tiles[pos]= temp3
    
    test_time= list(df_test['OS time'])
    test_event= list(df_test['Event'])
    
    test_x=[load_img(j, target_size=(image_dim, image_dim)) for i in (test_x_tiles) for j in i] 
    test_x = [image.img_to_array(i) for i in test_x]
    test_x = np.asarray(test_x)
    test_x= test_x/255.0
    
    test_time= np.asarray(test_time)
    test_time= np.repeat(test_time, num_patches)
    test_event= np.asarray(test_event)
    test_event= np.repeat(test_event, num_patches)
    
    train_fn = InputFunction(train_x, train_time, train_event, batch_size=64, drop_last = True, shuffle = True)
    eval_fn = InputFunction(test_x, test_time, test_event, batch_size=64, drop_last = True, shuffle = False)
    
    base_model= ResNet101(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions= Dense(1, activation='linear')(x) #=>best cindex is 0.66
    
    model = Model(inputs=base_model.input, outputs=predictions)
    trainer = TrainAndEvaluateModel(
            model = model,
            model_dir=Path("ckpts-mnist-cnn"),
            train_dataset = train_fn(),
            eval_dataset = eval_fn(),
            learning_rate = 1e-4,
            num_epochs = 25, #15
        )
    trainer.train_and_evaluate()
    


