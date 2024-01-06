import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.regularizers import L2
#from tensorflow.math import exp, sqrt, square

#credit: the majority of this code is copied from assignment 5 (VAEs), the base code was written by course staff and the rest filled in 
#by Sam Musker in his assignment submission. 

class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=2, latent_size=15, learning_rate=0.01/20):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_classes = num_classes 
        self.hidden_dim = 500 #0 #was 1000, 5000 #this is hidden dim for encoder
        self.hidden_dim_dec = 5000
        self.lr = learning_rate
        
        self.encoder = tf.keras.Sequential()
        self.mu_layer = tf.keras.layers.Dense(self.latent_size, input_shape=(self.hidden_dim,)) 
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size, input_shape=(self.hidden_dim,))
        self.decoder = tf.keras.Sequential() 

        self.adamoptimizer=tf.keras.optimizers.Adam(self.lr) #was 0.01/10

        #these are the encoder layers

        #conv2d
        self.encoder.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28,28,1), activation='relu'))
#         self.encoder.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28,28,1), activation='relu'))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.MaxPool2D())
        self.encoder.add(tf.keras.layers.Dropout(0.5))

        self.encoder.add(tf.keras.layers.Flatten())

        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, kernel_regularizer=L2(l2=0.01), input_shape=(self.input_size,), activation='relu'))
        self.encoder.add(tf.keras.layers.Dropout(0.5))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, kernel_regularizer=L2(l2=0.01),input_shape=(self.hidden_dim,), activation='relu'))
        self.encoder.add(tf.keras.layers.Dropout(0.5))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim, kernel_regularizer=L2(l2=0.01),input_shape=(self.hidden_dim,), activation='relu'))

        #this is where the classifier branches off
        self.c1 = (tf.keras.layers.Dense(self.num_classes, input_shape=(self.hidden_dim,), activation='softmax'))

        #these are the decoder layers

        self.decoder.add(tf.keras.layers.Dense(self.hidden_dim_dec, input_shape=(self.latent_size+num_classes,), kernel_regularizer=L2(l2=0.01), activation='relu'))
        self.decoder.add(tf.keras.layers.Dropout(0.5))
        self.decoder.add(tf.keras.layers.Dense(self.hidden_dim_dec, input_shape=(self.hidden_dim_dec,), kernel_regularizer=L2(l2=0.01), activation='relu'))
        self.decoder.add(tf.keras.layers.Dense(self.hidden_dim_dec, input_shape=(self.hidden_dim_dec,), kernel_regularizer=L2(l2=0.01), activation='relu'))
        self.decoder.add(tf.keras.layers.Dense(self.input_size, kernel_regularizer=L2(l2=0.01), input_shape=(self.hidden_dim_dec,), activation='sigmoid'))

        self.decoder.add(tf.keras.layers.Reshape((-1, 28, 28)))


    def call(self, x, c=None, only_classifier=False):
        x_hat = None
        mu = None
        logvar = None

        flattening_layer = Sequential([
            Flatten()
        ])
        x=flattening_layer(x)

        x=tf.reshape(x,[-1,28,28,1])
        encoder_outs=self.encoder(x)

        classifier_outputs = self.c1(encoder_outs)

        if (not(only_classifier)): #if it's called by test, we want only the classifier to run as we don't have a c. 
            mu=self.mu_layer(encoder_outs)
            logvar=self.logvar_layer(encoder_outs)

            z=reparametrize(mu,logvar)
            zc=tf.concat([z,c],axis=1)
            #x_hat=self.d8(self.d7(self.d6(self.d5(self.d4(self.d3(self.d2(self.d1(zc))))))))
            x_hat=self.decoder(zc)

        return x_hat, mu, logvar, classifier_outputs


def reparametrize(mu, logvar):
    z = None

    epsilon=tf.random.normal(shape=mu.shape)

    z=mu+(((tf.exp(logvar))**0.5)*epsilon)

    return z

def bce_function(x_hat, x):

    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, 
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[-1] 
    return reconstruction_loss

#classifier helper functions start

def classifier_loss(probs, labels, batch_size):
    """
    Calculates loss for each batch during training
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    losses = loss_fn(y_true=labels, y_pred=probs)
    return tf.math.reduce_sum(losses) / batch_size

#classifier helper functions end

def loss_function(x_hat, x, mu, logvar, classifier_outs, one_hot_labels, batch_size):

    cvae_loss = None

    sigsq=tf.exp(logvar)
    dk_loss=(-0.5)*tf.reduce_sum(1+logvar-(tf.square(mu))-sigsq)
    rec_loss=bce_function(x_hat, x)

    cvae_loss=rec_loss+dk_loss

    cvae_loss=cvae_loss/x_hat.shape[0]

    class_loss = classifier_loss(classifier_outs, one_hot_labels, batch_size)

    cvae_loss_weight = 1

    classifier_loss_weight = 100000

    return ((cvae_loss*cvae_loss_weight)+(class_loss*classifier_loss_weight))
