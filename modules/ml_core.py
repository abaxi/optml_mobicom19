import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation

from keras.optimizers import RMSprop, Adam, Nadam

from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

def get_model(in_dim, out_dim, h_units, num_h_layers, act_hid, optimizer, loss_fn, act_in=None, act_out=None, learning_rate=0.01):
    '''
    in_dim = size of input vector, i.e. channel from a single antenna. 
    out_dim = size of output vector of NNDE
    num_h_layers = number of hidden layers
    act_hid = activation function for hidden layers, "elu/relu/..."
    optimizer = adam/nadam/etc... if invalid parameter value is passed 
                then rmsprop is default
    loss_fn
    act_in/act_out = activation function for input/output layer, defaults to act_hid
    learning_rate = rate for optimizer, defaults to 0.01
    '''
    
    if act_in == None:
        act_in = act_hid
    if act_out == None:
        act_out = act_hid
    
    model = Sequential()
    for i in range(num_h_layers):
        if i == 0:
            ### first layer needs extra info
            model.add(Dense(units=h_units, input_dim=in_dim))
            model.add(Activation(act_in))
        else:
            ### all hidden layers
            model.add(Dense(units=h_units))
            model.add(Activation(act_hid))
    
    ### last or output layer
    model.add(Dense(units=out_dim))
    model.add(Activation(act_out))
        
    if optimizer == "adam":
        optimizer = Adam(lr=learning_rate)
    elif optimizer == "nadam":
        optimizer = Nadam(lr=learning_rate)
    else:
        optimizer = RMSprop(lr=learning_rate)

    model.compile(optimizer, loss=loss_fn, metrics=[loss_fn])
    return model


def get_callbacks(es_th=None, file_name=None, pat=None, min_del=None):
    '''
    es_th = early stopping threshold. once loss is below this threshold, training will stop.
    file_name = best checkpointed keras NN model will be written to disk with name <file_name>
    pat = patience for early stopping. number of iterations within which min_del improvement must be seen, 
            otherwise training stops. pat and min_del must both be set for early stopping due to low improvement
    min_del = see: pat. 
    '''
    callbacks_list = []
    if es_th !=None:
        early_stop_low_loss = EarlyStoppingByLossVal(monitor='val_loss', value=es_th, verbose=1)
        callbacks_list.append(early_stop_low_loss)
    
    if min_del!=None and pat!=None:
        early_stop_no_gain = EarlyStopping(monitor='val_loss', min_delta=min_del, patience=pat, verbose=1, mode='auto')
        callbacks_list.append(early_stop_no_gain)
    
    if len(file_name)!=0:
        save_best = ModelCheckpoint(filepath=file_name, verbose=0, save_best_only=True, save_weights_only=False, period=1)
        callbacks_list.append(save_best)
    
    return callbacks_list

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.1, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
