import os
import tensorflow as tf
import numpy as np
import random
import math

def get_data(path, rows=None, random_sample=None, categories=None):
    """"
    Grabs data from files
    path = directory of .npy files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
    row = optional, number of images we want from each class. Default grabs all the images from file
    random_sample = optional, how many categories to randomly sample from
    categories = optional, list of category names to use
    returns:
    - all_data = features of size 28x28 (images)
    - labels = one hot encoded labels
    - class_mapping = class_name -> one_hot encoding index dictionary
    """
    num_classes = 0
    all_data = np.empty((0, 28, 28))
    labels = []
    class_mapping = {}
    
    files_to_use = []
    all_files = os.listdir(path)
    
    if random_sample is not None:
        files_to_use = random.sample(all_files, random_sample)
        
    if categories is not None:
        for c in categories:
            file_name = f'full_numpy_bitmap_{c}.npy'
            to_find = path+file_name
            if os.path.exists(to_find):
                files_to_use.append(file_name)

    if random_sample == None and categories == None:
        for filename in os.listdir(path):
            if filename.endswith('.npy'):
                files_to_use.append(filename)
                
    for file in files_to_use:
        data = np.load(os.path.join(path, file))
        data = data / 255.0
        data = np.reshape(data, (-1, 28, 28))

        shuffle_indices = tf.random.shuffle(np.arange(data.shape[0]))[:rows]
        data = tf.gather(data, shuffle_indices)

        labels += [num_classes] * data.shape[0]
        class_name = (file.split('_')[-1]).split('.')[0]
        class_mapping[num_classes] = class_name
        num_classes +=1

        all_data = np.vstack( [all_data, data] )

    labels = tf.one_hot(np.array(labels, dtype=np.int32), num_classes)
    return all_data, labels, class_mapping, num_classes


def split_dataset(inputs, labels, test_size=0.2):
    """
    Splits dataset from get_data into train/test
    inputs = all_data
    labels = labels
    test_size = size of testing dataset
    returns:
    - X_train = training features
    - X_test = testing features
    - y_train = training labels
    - y_test = testing labels
    """
    num = inputs.shape[0]
    indices = np.random.permutation(num)
    bounds = num*(1-test_size)
    train_ids = indices[:int(np.ceil(bounds))]
    test_ids = indices[int(np.floor(bounds)):]
    X_train, X_test = tf.gather(inputs, train_ids), tf.gather(inputs, test_ids)
    y_train, y_test = tf.gather(labels, train_ids), tf.gather(labels, test_ids)
    X_train = np.reshape(X_train, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))
    return X_train, X_test, y_train, y_test

