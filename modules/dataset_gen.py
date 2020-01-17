from multiprocessing import Pool
import numpy as np
import random
import bisect
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
import RF_common

def to_complex(chans):
    '''
    takes in real channels, gets them back to complex 
    valued by reversing the processing in fn to_reals()
    '''
    if chans.shape[1]%2 != 0:
        print "ERROR:cannot convert odd #cols to complex"
        return None
    mid = chans.shape[1]/2
    real = chans[:,:mid]
    real = np.array(real)
    imag = chans[:,mid:]
    imag = np.array(imag)
    complex_chan = real+1j*imag
    return complex_chan

def to_reals(chans):
    '''
    takes in complex channels, converts them to real valued:
    [a+ib, c+id] --> [a,b,c,d]
    '''
    if len(chans.shape)==2:
        num_chans = chans.shape[0]
        mid = chans.shape[1]
        X2 = np.zeros([num_chans,2*mid])
        X2[:,:mid] = np.real(chans)
        X2[:,mid:] = np.imag(chans)
        X2 = np.array(X2)
    else:
        mid = chans.shape[0]
        X2 = np.zeros([2*mid,1])
        X2[:mid,0] = np.real(chans)
        X2[mid:,0] = np.imag(chans)
    return X2

def get_params_multi_proc(n_chans, max_n_paths, max_d, n_processes, min_d, min_n_paths, sep):
    '''
    helper function to speed up channel parameter generation.
    multiple processors are used and their results are combined.
    '''
    p = Pool(processes=n_processes)
    num_chans = n_chans/n_processes
    results = []
    for i in range(n_processes):
        results.append(p.apply_async(get_params, args=(num_chans, max_n_paths, max_d, min_d, min_n_paths, sep)))
    p.close()
    p.join()
    output = [r.get() for r in results]
    params_list = []
    for i in range(0,len(output)):
        x = output[i]
        params_list.extend(x)
    return params_list

def get_params(n_chans, max_n_paths, max_d, min_d, min_n_paths, sep):
    params = []
    np.random.seed()
    for i in range(n_chans):
        nps = np.random.randint(min_n_paths,max_n_paths+1)
        if sep !=None:
            d_ns, a_ns, phi_ns, psi_ns = RF_common.get_synth_params_sep(nps, min_d, max_d, sep)
        else:
            d_ns, a_ns, phi_ns, psi_ns = RF_common.get_synth_params(nps, min_d, max_d)

        p = np.array(d_ns), np.array(a_ns), np.array(phi_ns), np.array(psi_ns)
        params.append(p)
    return params

def get_array_chans_multi_proc(lambs, K, sep, params_list, n_processes, norm):
    '''
    helper function to speed up generation of channels based on parameters.
    multiple processors are used and their results are combined.
    '''
    p = Pool(processes=n_processes)
    num_chans = len(params_list)
    num_chans_per_proc = num_chans/n_processes
    results = []
    for i in range(n_processes):
        start = i*num_chans_per_proc
        end = min(start+num_chans_per_proc, num_chans)
        sub_params_list = params_list[start:end]
        results.append(p.apply_async(get_array_chans, args=(lambs,K, sep, sub_params_list, norm)))
    p.close()
    p.join()
        
    output = [p.get() for p in results]
    X = None
    for i in range(0,len(output)):
        x = output[i]
        if i == 0:
            X = x
        else:
            X = np.vstack([X,x])
    del results, output
    return X

def get_array_chans(l1, K, sep, params_list, norm):
    chans = []
    num_chans = len(params_list)
    for i in range(num_chans):
        params = params_list[i]
        x = RF_common.get_chans_from_params(params, K, sep, l1)
        x = x.transpose()
        x = x.ravel()
        if norm:
            x = x/np.max(np.abs(x))
        chans.append(x)
    chans = np.array(chans)
    return chans

'''
code for generating 1D outputs, 
storing them in sparse format to save space using get_sparse_target(),
decompressing on the fly when training using nn_batch_generator_convolved()
'''
def get_sparse_target(params_list, distances, lowest):
    nrows = len(params_list)
    ncols = len(distances)+1
    sparse_data = []
    sparse_i = []
    sparse_j = []
    for rowID in range(nrows):
        params = params_list[rowID]
        dns = params[0]
        ans = params[1]
        n_paths = len(dns)
        for i in range(n_paths):
            d = dns[i]
            a = max(ans[i], lowest)
            pos = bisect.bisect_left(distances, d)
            sparse_data.append(a)
            sparse_i.append(rowID)
            sparse_j.append(pos)
    Y = csr_matrix((sparse_data, (sparse_i, sparse_j)), shape=(nrows, ncols))
    return Y

def nn_batch_generator_convolved(X_dense, y_sparse, batch_size, conv_filter):
    '''
    a generator function that can be fed to model.fit_generator
    useful if y is extremly large but sparse
    '''
    samples_per_epoch = X_dense.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(X_dense)[0])

    ncols = y_sparse.shape[1]

    while 1:
        batch_indexes = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_dense[batch_indexes,:]
        y_batch = np.array(y_sparse[batch_indexes,:].todense())
        y_batch_op = np.zeros([y_batch.shape[0], ncols])

        # full_len = y_batch.shape[1] + len(conv_filter)
        # start = full_len/2 - y_batch.shape[1]/2
        # end = start + y_batch.shape[1]

        for i in range(y_batch.shape[0]):
            # temp = np.convolve(y_batch[i,:], conv_filter, "full")[start:end]
            temp = gaussian_filter(y_batch[i,:], conv_filter)
            temp = temp/np.max(temp)
            # temp = np.max(y_batch[i,:])*temp
            y_batch_op[i,:] = temp

        counter += 1
        yield np.array(X_batch),y_batch_op
        if (counter > number_of_batches):
            counter=0
