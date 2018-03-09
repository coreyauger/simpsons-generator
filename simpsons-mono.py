import os

import numpy as np
from scipy import misc

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop, Adam

import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.interactive(True)
matplotlib.use('Agg')

img_w, img_h = 64, 64
img_channels = 3

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_nparray_from_data():
    with open("./data/annotation.txt") as f:
        content = f.readlines()
    content = ["./data/" + x.strip().split(",")[0].replace("./","") for x in content] 
    img_test = content[ int(np.random.random() * len(content)) ] 
    data = np.array( [misc.imresize(crop_center(rgb2gray(misc.imread(x)), 256, 256), (img_w, img_h) ) for x in content] )
    data = np.reshape(data, [data.shape[0], img_w, img_h, 1])
    data = data/255
    return data

#np.save("./data/data-mono.npy",create_nparray_from_data())
#data = np.load("./data/data-small.npy")
data = np.load("./data/data-mono.npy")
print(data.shape)

img_test = data[ int(np.random.random() * data.shape[0]),:,:,0 ] 
plt.imshow(img_test, cmap="gray")
plt.show()
print(img_test)

input("Press Enter to start training.")


def discriminator_builder(depth=64,p=0.4):
    
    # Define inputs
    inputs = Input((img_w,img_h,1))
    
    # Convolutional layers
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)
    
    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)
    
    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)
    
    conv4 = Conv2D(depth*8, 5, strides=2, padding='same', activation='relu')(conv3)
    conv4 = Dropout(p)(conv4)

    conv5 = Conv2D(depth*16, 5, strides=1, padding='same', activation='relu')(conv4)
    conv5 = Flatten()(Dropout(p)(conv5))
    
    output = Dense(1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

discriminator = discriminator_builder()
discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])

def generator_builder(z_dim=100,depth=64,p=0.4):
    
    # Define inputs
    inputs = Input((z_dim,))
    
    # First dense layer
    dense1 = Dense(16*16*64)(inputs)
    dense1 = BatchNormalization(axis=-1,momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((16,16,64))(dense1)
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

    #conv4 = UpSampling2D()(conv2)
    conv4 = Conv2DTranspose(int(depth/16), kernel_size=5, padding='same', activation=None,)(conv3)
    conv4 = BatchNormalization(axis=-1,momentum=0.9)(conv4)
    conv4 = Activation(activation='relu')(conv4)
    # Define output layers
    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv4)

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


def train(epochs=2000,batch=128):
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    for i in range(epochs):
        real_imgs = np.reshape(data[np.random.choice(data.shape[0],batch,replace=False)],(batch,img_w,img_h,1))
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))
        x = np.concatenate((real_imgs,fake_imgs)) 
        y = np.ones([2*batch,1])
        y[batch:,:] = 0
        make_trainable(discriminator, True)
        d_loss.append(discriminator.train_on_batch(x,y))
        if False:
            plt.figure(figsize=(5,5))
            for k in range(16):
                plt.subplot(4, 4, k+1)
                plt.imshow(real_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            input("Press Enter to start training.")
            img_test = real_imgs[ int(np.random.random() * real_imgs.shape[0]),:,:,0 ] 
            plt.imshow(img_test, cmap="gray")
            plt.show()
            print(img_test)
            input("Press Enter to start training.")
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
            plt.figure(figsize=(5,5))
            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            #plt.show()
            plt.savefig('./images/mono_{}.png'.format(i+1))
        if (i+1)%10000 == 0:
            AM.save("./data/model_mono_{}.h5".format(i+1))
            print("Saved model..")
    return a_loss, d_loss
        
a_loss, d_loss = train(epochs=250000)
