# -*- coding: utf-8 -*-
"""scheduledCC_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LUDWDZsHYAYHxnieK9agxXj0-nSODlGA
"""

import os
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')))

import pickle
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import functions

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 123
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def plot_final_graph(addition="", third=True, ylim_loss=4, ylim_error=1):
    history_no_clipping = pickle.load(open(addition + 'trainHistoryDict', "rb"))
    history_05 = pickle.load(open(addition + 'trainHistoryDict_clip_05', "rb"))
    if third:
        history_1 = pickle.load(open(addition + 'trainHistoryDict_clip_1', "rb"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].grid(True)
    axs[1].grid(True)
    max_len = len(history_no_clipping['val_loss'])
    axs[0].plot(history_no_clipping['val_loss'][4:max_len:5], label='no clipping')
    axs[0].plot(history_05['val_loss'][4:max_len:5], label='0.5')
    if third:
        axs[0].plot(history_1['val_loss'][4:max_len:5], label='1')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylim(0, ylim_loss)
    axs[1].plot(1 - np.array(history_no_clipping['val_acc'][4:max_len:5]), label='no clipping')
    axs[1].plot(1 - np.array(history_05['val_acc'][4:max_len:5]), label='0.5')
    if third:
        axs[1].plot(1 - np.array(history_1['val_acc'][4:max_len:5]), label='1')
    axs[1].set_title('Error')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylim(0, ylim_error)
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')

"""### Loading Data"""

num_classes = 10

# load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions.
input_shape = x_train.shape[1:]

# normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# if subtract pixel mean is enabled
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

"""### model = 3 (accroding to "CircConv" article)"""

schedule3 = [2] * 16
schedule3[:3] = [1] * 3
schedule3[6], schedule3[11] = 1, 1

save_dir = "C:/source/ml/"

"""#### Model without clipping"""

model, model_type = functions.define_and_compile_ResNet_model(
    input_shape, conv_layer = functions.CircConv2D, n=2, compress_schedule=schedule3)
model.summary()

history = functions.run_training(model, model_type, x_train, y_train, x_test, y_test,
                       'n_2_sch_3_trainHistoryDict', steps_per_epoch=100, batch_size=100,
                       epochs=10)

history = pickle.load(open('n_2_sch_3_trainHistoryDict', "rb"))
functions.plot_loss_acc(history, 2, 0.6)
plt.savefig(save_dir + "n_2_sch_3_plot.pdf", format='pdf')

with open(save_dir + 'n_2_sch_3_trainHistoryDict', 'wb') as f:
    pickle.dump(history, f)

"""#### Model with clipping to 0.5"""

model, model_type = functions.define_and_compile_ResNet_model(
    input_shape, conv_layer = functions.CircConv2D, n=2, compress_schedule=schedule3)

callbacks = functions.standard_callbacks(model_type) + [functions.Clipping(0.5)]
history = functions.run_training(model, model_type, x_train, y_train, x_test, y_test,
                       'n_2_sch_3_trainHistoryDict_clip_05', steps_per_epoch=100, epochs=10,
                       batch_size=100,
                       callbacks=callbacks)

history = pickle.load(open('n_2_sch_3_trainHistoryDict_clip_05', "rb"))
functions.plot_loss_acc(history, 2, 0.6)
plt.savefig(save_dir + "n_2_sch_3_clip_05_plot.pdf", format='pdf')

with open(save_dir + 'n_2_sch_3_trainHistoryDict_clip_05', 'wb') as f:
    pickle.dump(history, f)

"""#### Model with clipping to 1"""

model, model_type = functions.define_and_compile_ResNet_model(
    input_shape, conv_layer = functions.CircConv2D, n=2, compress_schedule=schedule3)

callbacks = functions.standard_callbacks(model_type) + [functions.Clipping(1)]
history = functions.run_training(model, model_type, x_train, y_train, x_test, y_test,
                       'n_2_sch_3_trainHistoryDict_clip_1', steps_per_epoch=100, epochs=10,
                       batch_size=100,
                       callbacks=callbacks)

history = pickle.load(open('n_2_sch_3_trainHistoryDict_clip_1', "rb"))
functions.plot_loss_acc(history, 2, 0.6)
plt.savefig(save_dir + "n_2_sch_3_clip_1_plot.pdf", format='pdf')

with open(save_dir + 'n_2_sch_3_trainHistoryDict_clip_1', 'wb') as f:
    pickle.dump(history, f)

"""#### Plotting final graph"""

plot_final_graph('n_2_sch_3', ylim_loss=2.5, ylim_error=0.3)
plt.savefig(save_dir + "n_2_sch_3_clip_1_plot_final.pdf", format='pdf')
