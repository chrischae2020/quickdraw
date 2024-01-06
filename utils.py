import tensorflow as tf
import numpy as np
from cvae import CVAE, reparametrize, loss_function


def classifier_accuracy(pred, labels):
    """
    Accuracy for testing across batch
    """
    correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def test(model, inputs, labels):
    acc_list = []
    for i in range(0, len(inputs), 128):
        x = inputs[i:i+128]
        y = labels[i:i+128]
        _, _, _, pred = model.call(x, only_classifier=True) #modified to use the cvae call - fourth output is classifier
        test_acc = classifier_accuracy(pred, y)
        acc_list.append(test_acc)
    return sum(acc_list) / len(acc_list)

def validation(model, inputs, labels, batch_size):
    sum_losses = 0
    for i in range(0, len(inputs), batch_size):
        x = inputs[i:i+batch_size]
        y = labels[i:i+batch_size]
        generated_egs, mu, logvar, classifier_outs =model.call(x,y)
        loss=loss_function(generated_egs,x,mu,logvar,classifier_outs, y, batch_size)
        sum_losses+=loss
    return sum_losses

#credit: the majority of this code is copied from assignment 5 (VAEs), the base code was written by course staff and the rest filled in 
#by Sam Musker in his assignment submission. 

def train_cvae(model, inputs, labels, batch_size):
    shuffle_indices = tf.random.shuffle(np.arange(inputs.shape[0]))
    train_inputs = tf.gather(inputs, shuffle_indices)
    train_inputs = tf.image.random_flip_left_right(train_inputs)
    train_labels = tf.gather(labels, shuffle_indices)

    sum_losses=0

    num_batches=int(np.ceil(inputs.shape[0]/batch_size))

    for i in range(0, len(inputs), batch_size): 
        with tf.GradientTape() as tape:
            batch_inputs=inputs[i:i+batch_size]

            batch_labels_oh=labels[i:i+batch_size]
            
            generated_egs, mu, logvar, classifier_outs =model.call(batch_inputs,batch_labels_oh)

            loss=loss_function(generated_egs,batch_inputs,mu,logvar,classifier_outs, batch_labels_oh, batch_size)

            sum_losses+=loss
        
        model.adamoptimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    return sum_losses
