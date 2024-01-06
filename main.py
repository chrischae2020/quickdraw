#from cnn import Model
from preprocess import get_data, split_dataset
from utils import test, train_cvae, validation #train, test, train_cvae
from cvae import CVAE, reparametrize, loss_function

import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
#from tensorflow.math import sigmoid #original
#from tensorflow import sigmoid
from tqdm import tqdm

import argparse
import os
import sys
import time
from inputs import image_to_np
from datetime import datetime

parser = argparse.ArgumentParser(description='QuickDrawCVAE')

parser.add_argument('--img_data', type=str, default='./data/', help='training image data directory (must consist of .npy files)')
parser.add_argument('--num_imgs', type=int, default=None, help='number of rows for training/testing')
parser.add_argument('--latent_size', type=int, default=300, help='latent size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--random_sample', type=int, default=None, help='Number of categories to randomly sample from, do not mix with categories')
parser.add_argument('--categories', nargs='+', default=None, help='categories to train model, do not mix with random_sample')
parser.add_argument('--save_outputs', type=bool, default=False, help='save output images from CVAE and loss curves')
parser.add_argument('--outputs_dir', type=str, default='./outputs/', help='output dir from images produced by CVAE')
parser.add_argument('--device', type=str, default='GPU' if tf.test.is_gpu_available else 'CPU', help='device')

args = parser.parse_args()

#credit: the show_cvae_images code is copied from assignment 5 (VAEs), the code was written by course staff

def show_cvae_images(model, latent_size,nc, output_dir):

    num_generation = 100
    num_classes = nc
    num_per_class = num_generation // num_classes
    c = tf.eye(num_classes)
    z = []
    labels = []
    for label in range(num_classes):
        curr_c = c[label]
        curr_c = tf.broadcast_to(curr_c, [num_per_class, len(curr_c)])
        curr_z = tf.random.normal(shape=[num_per_class,latent_size])
        curr_z = tf.concat([curr_z,curr_c], axis=-1)
        z.append(curr_z)
        labels.append([label]*num_per_class)
    z = np.concatenate(z)
    labels = np.concatenate(labels)
    samples = model.decoder(z).numpy()

    rows = num_classes
    cols = num_generation // rows

    fig = plt.figure(figsize=(cols, rows))
    gspec = gridspec.GridSpec(rows, cols)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    os.makedirs(output_dir, exist_ok=True)
    file_name = output_dir+f'cvae_images_img{args.num_imgs}_ls_{args.latent_size}_bs_{args.batch_size}.pdf'
    print(f'saved cvae images as {file_name}')
    
    plt.savefig(file_name, bbox_inches="tight")
    plt.close(fig)

def plot_loss(running_train_loss, running_val_loss):
    fig, axes = plt.subplots(1,1, figsize = (9, 9))
    axes.plot(running_train_loss, label='Train')
    axes.plot(running_val_loss, label="Val")
    axes.legend()
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Loss')
    file_name = args.outputs_dir+f'loss_curve_img{args.num_imgs}_ls_{args.latent_size}_bs_{args.batch_size}.jpg'
    print(f'saved loss curve as {file_name}')
    fig.savefig(file_name)
    plt.close(fig)

def plot_acc(running_train_acc, running_val_acc):
    fig, axes = plt.subplots(1,1, figsize = (9, 9))
    axes.plot(running_train_acc, label='Train')
    axes.plot(running_val_acc, label="Val")
    axes.legend()
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Accuracy')
    file_name = args.outputs_dir+f'acc_curve_img{args.num_imgs}_ls_{args.latent_size}_bs_{args.batch_size}.jpg'
    print(f'saved acc curve as {file_name}')
    fig.savefig(file_name)
    plt.close(fig)
    
def main():
    data_path = args.img_data
    data, labels, mapping, num_classes_classifier = get_data(data_path, args.num_imgs, args.random_sample, args.categories)

    X_train, X_test, y_train, y_test = split_dataset(data, labels)

    X_train, X_val, y_train, y_val = split_dataset(X_train, y_train)
    now = datetime.now()

    print(f'training: {X_train.shape[0]} imgs -- validation: {X_val.shape[0]} imgs -- testing: {X_test.shape[0]} imgs')

    #begin CVAE portion

    ls=args.latent_size #was 15, changing to 200 seemed to make generated images a little worse

    input_size=28*28

    batch_size=args.batch_size

    nc=num_classes_classifier #num_classes
    print(f"total of {nc} classes")
    for k,v in mapping.items(): print(v)

    #we want inputs (N,1,H,W) and one hot labels (N,C). so squeeze and expand dims to put the dimension of size 1 in the right place. 

    X_train=tf.squeeze(X_train)

    X_train=tf.expand_dims(X_train,axis=1)
    
    X_val = tf.squeeze(X_val)
    X_val = tf.expand_dims(X_val,axis=1)

    running_train_loss = []
    running_val_loss = []
    running_train_acc = []
    running_val_acc = []

    with tf.device(args.device):
    
        cvae_model = CVAE(input_size, num_classes=nc, latent_size=ls) #input_size, latent_size

        for epoch_id in range(1,args.epochs+1): #num_epochs, was 10
            print(f"===== Epoch {epoch_id} =====")
            start = time.time()
            train_loss = train_cvae(cvae_model, X_train, y_train, batch_size)/X_train.shape[0] #inputs, labels, batch_size
            val_loss = validation(cvae_model, X_val, y_val, batch_size)/X_val.shape[0]
            train_acc = np.array(test(cvae_model, X_train, y_train))
            val_acc = np.array(test(cvae_model, X_val, y_val))
            end = time.time()
            print(f"Train Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Acc: {val_acc:.4f}\tElapsed: {end-start:.3f}s")
            running_train_loss.append(train_loss)
            running_val_loss.append(val_loss)
            running_train_acc.append(train_acc)
            running_val_acc.append(val_acc)

        if args.save_outputs:
            show_cvae_images(cvae_model, ls, nc, args.outputs_dir)
            plot_loss(running_train_loss, running_val_loss)
            plot_acc(running_train_acc, running_val_acc)

        #now test the classifier branch

        test_acc = test(cvae_model, X_test, y_test)
        
        save_model(cvae_model, mapping)
        
        print('test acc: ', np.array(test_acc)) 


def save_model(model,mapping):
    """
    Save trained VAE model weights to model_ckpts/

    Inputs:
    - model: Trained VAE model.
    - args: All arguments.
    """
    output_dir = os.path.join("model_ckpts", "model")
    output_path = os.path.join(output_dir, "weights")
    output_mapping = os.path.join(output_dir, "mapping")
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)
    with open(output_mapping, 'w') as mp:
        for item in mapping.values():
            mp.write(f"{item}\n")
    mp.close()


def load_model(model):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.
        
    Returns:
    - model: Trained model.
    """
    inputs = tf.zeros([1,1,28,28])  # Random data sample
    weights_path = os.path.join("model_ckpts", "model","weights")
    mapping_path = os.path.join("model_ckpts", "model","mapping")
    _ = model(inputs, c=None, only_classifier=True)
    model.load_weights(weights_path).expect_partial()
    mapping = []
    with open(mapping_path, 'r') as mp:
        for line in mp:
            x = line[:-1]
            mapping.append(x)
    mp.close()
    return model,mapping

def make_guess():
    model = CVAE(28*28, num_classes=10, latent_size=500) #WARNING: change these if the saved model has different configs
    model,mapping = load_model(model)
    drawing = image_to_np()
    _,_,_,probs = model.call(drawing, only_classifier=True)
    s = tf.argsort(probs[-1])
    guesses = tf.gather(mapping, s).numpy()
    print(f"top 3 guesses are: {guesses[:-4:-1]}, picking the first one")
    return guesses

def guess_was_right(guess):
    model = CVAE(28*28, num_classes=10, latent_size=500) #WARNING: change these if the saved model has different configs
    model,mapping = load_model(model)
    label = mapping.index(guess)
    num_per_class = 1
    c = tf.eye(10)  #WARNING: Edit if not using 10 categories
    z = []
    labels = []
    curr_c = c[label]
    curr_c = tf.broadcast_to(curr_c, [num_per_class, len(curr_c)])
    curr_z = tf.random.normal(shape=[num_per_class,500]) #EDIT if latent size is different
    curr_z = tf.concat([curr_z,curr_c], axis=-1)
    z.append(curr_z)
    labels.append([label]*num_per_class)
    z = np.concatenate(z)
    labels = np.concatenate(labels)
    samples = model.decoder(z).numpy()
    samples = 1-samples

    rows = 1
    cols = 1

    fig = plt.figure(figsize=(cols, rows))
    gspec = gridspec.GridSpec(rows, cols)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    file_name = 'generated.png'
    plt.savefig(file_name, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    main() 