import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

class Model(tf.keras.Model):
    def __init__(self, num_classes, batch_size=64):
        """
        Current model architecture
        - Conv2D
        - Batch normalization
        - Flatten
        - Dropout for 0.5
        - Dense w relu
        - Dense w softmax for probabilities
        """
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.loss_list = [] # for training

        self.seq = Sequential()
        self.seq.add(Conv2D(64, (3, 3), input_shape=((28,28,1)), padding='SAME', activation='relu'))
        self.seq.add(BatchNormalization())
        self.seq.add(Flatten())
        self.seq.add(Dropout(0.5))
        self.seq.add(Dense(120, activation='relu'))
        self.seq.add(Dense(self.num_classes, activation='softmax'))

        self.lr = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def call(self, inputs):
        return self.seq(inputs)

    def loss(self, probs, labels):
        """
        Calculates loss for each batch during training
        """
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        losses = loss_fn(y_true=labels, y_pred=probs)
        return tf.math.reduce_sum(losses) / self.batch_size

    def accuracy(self, pred, labels):
        """
        Accuracy for testing across batch
        """
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))