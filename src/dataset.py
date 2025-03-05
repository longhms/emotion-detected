from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
import tensorflow as tf

DATA_OUTPUT_PATH = '/kaggle/working/dataset/'
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
IMAGE_DEPTH = 1
CLASS_LABELS = [2, 3, 4, 5, 6]
CLASS_NAMES = ['Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype=int)
    # Assuming images are grayscal
    return pixels.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))


def create_generators(root_dir ,batch_size=64):
    
    df = pd.read_csv(root_dir)
    
    train_dir = os.path.join(DATA_OUTPUT_PATH, 'train/')
    validation_dir = os.path.join(DATA_OUTPUT_PATH, 'val/')
    test_dir = os.path.join(DATA_OUTPUT_PATH, 'test/')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']

    train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(validation_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)
    
    train_dataFrame = pd.read_csv(os.path.join(DATA_OUTPUT_PATH, 'train/train.csv'))
    val_dataFrame = pd.read_csv(os.path.join(DATA_OUTPUT_PATH, 'val/val.csv'))
    test_dataFrame = pd.read_csv(os.path.join(DATA_OUTPUT_PATH, 'test/test.csv'))
    
    train_dataFrame = train_dataFrame[train_dataFrame.emotion.isin(CLASS_LABELS)]
    val_dataFrame = val_dataFrame[val_dataFrame.emotion.isin(CLASS_LABELS)]
    test_dataFrame = test_dataFrame[test_dataFrame.emotion.isin(CLASS_LABELS)]
    
    train_labels = tf.keras.utils.to_categorical(
        train_dataFrame['emotion'].apply(lambda x: x-2))
    val_labels = tf.keras.utils.to_categorical(
        val_dataFrame['emotion'].apply(lambda x: x-2))
    test_labels = tf.keras.utils.to_categorical(
        test_dataFrame['emotion'].apply(lambda x: x-2))
    
    train_dataFrame['pixels'] = train_dataFrame['pixels'].apply(preprocess_pixels)
    val_dataFrame['pixels'] = val_dataFrame['pixels'].apply(preprocess_pixels)
    test_dataFrame['pixels'] = test_dataFrame['pixels'].apply(preprocess_pixels)
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        zca_whitening=False,
    )
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    
    train_gen = train_datagen.flow(
        x=np.array(train_dataFrame['pixels'].tolist()), 
        y=train_labels,
        batch_size=batch_size
    )
    
    val_gen = validation_datagen.flow(
        x=np.array(val_dataFrame['pixels'].tolist()),  
        y=val_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_gen = test_datagen.flow(
        x=np.array(test_dataFrame['pixels'].tolist()), 
        y=test_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen, test_labels
