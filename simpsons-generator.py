import os

import numpy as np

import keras
from scipy import misc
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop

import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
matplotlib.interactive(True)

img_w, img_h = 256, 256
img_channels = 3

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def create_nparray_from_data():
    with open("./data/annotation.txt") as f:
        content = f.readlines()
    content = ["./data/" + x.strip().split(",")[0].replace("./","") for x in content] 
    img_test = content[ int(np.random.random() * len(content)) ] 
    x = misc.imread(img_test)
    plt.imshow(crop_center(x,img_w,img_h) )
    plt.show()
    data = np.array( [crop_center(misc.imread(x), img_w, img_h) for x in content] )
    data = data/255
    print(x.shape)
    return data

#np.save("./data/data.npy",create_nparray_from_data())
#data = np.load("./data/data-small.npy")
data = np.load("./data/data.npy")
print(data.shape)

img_test = data[ int(np.random.random() * data.shape[0]),:,:,: ] 
plt.imshow(img_test)
plt.show()

input("Press Enter to start training.")

def discriminator_builder(depth=64, p=0.4):
    
    # Define inputs
    inputs = Input((img_w,img_h,img_channels))    
    
    # Convolutional layers
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu', input_shape=(img_w,img_h,img_channels))(inputs)
    conv1 = Dropout(p)(conv1)
    
    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)
    
    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)
    
    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))
    
    output = Dense(1, activation='sigmoid')(conv4)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

discriminator = discriminator_builder()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])

def generator_builder(z_dim=100,depth=64,p=0.4):
    
    # Define inputs
    inputs = Input((z_dim,))
    
    # First dense layer
    dense1 = Dense(64*64*256)(inputs)
    dense1 = BatchNormalization(axis=-1,momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((64,64,256))(dense1)
    dense1 = Dropout(p)(dense1)
    
    # Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(axis=-1,momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(axis=-1,momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)
    
    #conv3 = UpSampling2D()(conv2)
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(axis=-1,momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # Define output layers
    output = Conv2D(img_channels, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    # Model definition    
    model = Model(inputs=inputs, outputs=output)
    
    return model

generator = generator_builder()

def adversarial_builder(z_dim=100):    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8), metrics=['accuracy'])
    return model

AM = adversarial_builder()

def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for l in net.layers:
        l.trainable = is_trainable

def train(epochs=2000,batch=64):
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    for i in range(epochs):
        real_imgs = np.reshape(data[np.random.choice(data.shape[0], batch,replace=False)],(batch,img_w,img_h,img_channels))
        #print("real_imgs "+str(real_imgs.shape))
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))
        #print("fake_imgs "+str(fake_imgs.shape))
        x = np.concatenate((real_imgs,fake_imgs))
        y = np.ones([2*batch,1])
        y[batch:,:] = 0
        make_trainable(discriminator, True)
        d_loss.append(discriminator.train_on_batch(x,y))
        running_d_loss += d_loss[-1][0]
        running_d_acc += d_loss[-1][1]
        make_trainable(discriminator, False)
        
        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch,1])
        a_loss.append(AM.train_on_batch(noise,y))
        running_a_loss += a_loss[-1][0]
        running_a_acc += a_loss[-1][1]
        
        if (i+1)%10 == 0:
            print('Epoch #{}'.format(i+1))
        if (i+1)%250 == 0:
            print('Epoch #{}'.format(i+1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)
            print(log_mesg)
            noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
            gen_imgs = generator.predict(noise)
            print("gen_imgs: "+ str(gen_imgs.shape) )
            plt.figure()
            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs[k, :, :, :])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.savefig('./images/simpsons_{}.png'.format(i+1))
        if (i+1)%1000 == 0:
            AM.save("./data/model_{}.h5".format(i+1))
            print("Saved model..")
    return a_loss, d_loss
        
a_loss, d_loss = train(epochs=60000)

