import os, time, sys
import numpy as np
from modules import ml_core, RF_common, dataset_gen
import keras
import pylab as pl

def plot_1D_decompose_pred(Y_act, Y_pred, name):
    pl.figure(figsize=(15,10))
    for i in range(10):
        pl.subplot(5,2,i+1)
        x = Y_act[i][0]
        x = x.ravel().transpose()
        pl.plot(x)
        pl.plot(Y_pred[i])
        if i == 0:
            leg = ["Actual component positions", "Estimated"]
            pl.legend(leg)
    if name != "":
        pl.savefig(name)
        pl.close()
    else:
        pl.show()



if __name__ == '__main__':

    ### RF and antenna parameters
    sep = 0.15                          ## antenna separation, no effect when K=1
    nfft = 64                           ## nfft for channel
    K = 1                               ## num antennas
    cf = 2.4e9                          ## center freq
    bw = 10e6                           ## bandwidth over which channel is observed
    l1 = RF_common.get_lambs(cf, bw, nfft)  ## wavelengths, lambda for subcarriers in channel
    
    #### channel parameters
    min_d = 0                           ## min travel distance for any channel component, meters
    max_d = 100                          ## max travel distance for any channel component, meters
    nnde_step = 2                   ## for output of NNDE, the quantization step for distances, meters
    distances = np.arange(min_d, max_d, nnde_step) ## list of distances used for output of NNDE
    d_sep = None                           ## for generating channels, what is the min separation between any two components
    min_n_paths = 2                     ## min num paths
    max_n_paths = 3                     ## max num paths
    min_amp = 0.05                      ## min amplitude of any multipath component
    num_chans = 100000                    ## num of channels to generate
    n_cores = 2                         ## number of cores for parallelized data generation, minimum=1


    '''
    generate data for training and testing
    '''
    num_test = 2000     ### num channels for testing data set
    
    params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    te_params = dataset_gen.get_params_multi_proc(num_test, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)

    tr_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, True)
    te_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, te_params, n_cores, True)

    tr_X_complex = RF_common.add_noise_snr_range(tr_X_complex, 20, 30)
    te_X_complex = RF_common.add_noise_snr_range(te_X_complex, 20, 30)

    tr_X = dataset_gen.to_reals(tr_X_complex)
    te_X = dataset_gen.to_reals(te_X_complex)

    tr_Y = dataset_gen.get_sparse_target(params_list, distances, min_amp)
    te_Y = dataset_gen.get_sparse_target(te_params, distances, min_amp)

    data = tr_X, tr_Y, te_X, te_Y
    

    tr_X, tr_Y, te_X, te_Y = data
    print "I/O shapes:",tr_X.shape, tr_Y.shape, te_X.shape, te_Y.shape

    h_units = 200
    num_h_layers = 5
    sigma = 1.0
    
    cf_name = str(round(cf/1e9,2))      ## cf in string format for name
    bw_name = str(round(bw/1e6,2))      ## bw in string format for name
    name = cf_name+"_d"+str(max_d)+"_bw"+bw_name+"MHz"+"_1d_sigma_"+str(sigma)

    '''
    begin training NNDE
    '''
    ### NN parameters    
    activation = "elu"
    act_in = "elu"
    act_out = "elu"
    optimizer = "adam"
    lr = 0.001
    batch_size = 256
    num_epochs = 150
    loss_fn = loss_name = "mean_absolute_error"
    num_chans = tr_X.shape[0]
    steps_per_epoch = num_chans/batch_size
    in_dim = tr_X.shape[1]
    out_dim = tr_Y.shape[1]
    
    sigma = sigma*max(1,te_Y.shape[1]/200)
    d = str(te_Y.shape[1])
    conv_filter = sigma

    fname = name+".hdf5"
    callback_list = ml_core.get_callbacks(0.001, fname, 50, 0.01)
    print "setting:",name

    model = ml_core.get_model(in_dim, out_dim, h_units, num_h_layers, activation, optimizer, loss_name, act_in, act_out, learning_rate=lr)

    history = model.fit_generator(
        dataset_gen.nn_batch_generator_convolved(tr_X, tr_Y, batch_size, conv_filter),
        steps_per_epoch,
        epochs=num_epochs, 
        verbose=2,
        validation_data=dataset_gen.nn_batch_generator_convolved(te_X, te_Y, batch_size, conv_filter),
        validation_steps=1, 
        callbacks=callback_list
        )
    y_act = te_Y.todense()
    y_pred = model.predict(te_X)
    plot_1D_decompose_pred(y_act, y_pred, name+".png")

    keras.backend.clear_session()
    print "------"
