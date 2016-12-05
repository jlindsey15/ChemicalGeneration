"""
Specifies, trains and evaluates (cross-validation) neural fingerprint model.

Performs early-stopping on the validation set.

The model is specified inside the main() function, which is a demonstration of this code-base

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import time
import numpy as np
import sklearn.metrics as metrics

import warnings

import keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.optimizers as optimizers
import KerasNeuralfingerprint.utils as utils
import KerasNeuralfingerprint.data_preprocessing as data_preprocessing
import KerasNeuralfingerprint.fingerprint_model_matrix_based as fingerprint_model_matrix_based
import KerasNeuralfingerprint.fingerprint_model_index_based as fingerprint_model_index_based
from KerasNeuralfingerprint.fingerprint_model_matrix_based_predict import regression_frozen, generate_smiles
from sets import Set
from rdkit import Chem
from rdkit.Chem.MolSurf import _pyTPSA


import numpy as np

import types as python_types
import warnings
import copy
import os
import inspect
from six.moves import zip




def load_weights_by_name(model, filepath):
        '''Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        model.load_weights_from_hdf5_group_by_name(f)

        if hasattr(f, 'close'):
            f.close()

def load_weights_from_hdf5_group_by_name(model, f):
        ''' Name-based weight loading
        (instead of topological weight loading).
        Layers that have no matching name are skipped.
        '''
        if hasattr(model, 'flattened_layers'):
            # Support for legacy Sequential/Merge behavior.
            flattened_layers = model.flattened_layers
        else:
            flattened_layers = model.layers

        if 'nb_layers' in f.attrs:
                raise Exception('The weight file you are trying to load is' +
                                ' in a legacy format that does not support' +
                                ' name-based weight loading.')
        else:
            # New file format.
            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

            # Reverse index of layer name to list of layers with name.
            index = {}
            for layer in flattened_layers:
                if layer.name:
                    index.setdefault(layer.name, []).append(layer)

            # We batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weight_values = [g[weight_name] for weight_name in weight_names]

                for layer in index.get(name, []):
                    symbolic_weights = layer.weights
                    if len(weight_values) != len(symbolic_weights):
                        raise Exception('Layer #' + str(k) +
                                        ' (named "' + layer.name +
                                        '") expects ' +
                                        str(len(symbolic_weights)) +
                                        ' weight(s), but the saved weights' +
                                        ' have ' + str(len(weight_values)) +
                                        ' element(s).')
                    # Set values.
                    for i in range(len(weight_values)):
                        weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
            backend.batch_set_value(weight_value_tuples)


def lim(float, precision = 5):
    return ("{0:."+str(precision)+"f}").format(float)



def save_model_visualization(model, filename='model.png'):
    '''
    Requires the 'graphviz' software package
    '''



def predict(data, model):
    '''
    Returns a tensor containing the DNN's predictions for the given list of batches <data>
    '''
    pred = []    
    for batch in data:
        if len(batch)==3:
            batch = batch[0]
        pred.append(np.squeeze(model.predict_on_batch(batch)[0]))
    return np.concatenate(pred)



def eval_metrics_on(predictions, labels):
    labels = labels.flatten()

    '''
    assuming this is a regression task; labels are continuous-valued floats
    
    returns most regression-related scores for the given predictions/targets as a dictionary:
    
        r2, mean_abs_error, mse, rmse, median_absolute_error, explained_variance_score
    '''
    
    mean_abs_error           = np.abs(predictions - labels).mean()
    mse                      = ((predictions - labels)**2).mean()
    rmse                     = np.sqrt(mse)
    return {'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse}
    


def test_on(data, model, description='test_data score:'):
    '''
    Returns the model's mse on the given data
    '''
    scores=[]
    weights =[]
    for v in data:
        weights.append(v[1].shape) # size of batch
        loss, b, c, d = model.test_on_batch(x=v[0], y=[v[2], v[1]])
        scores.append(loss)
    weights = np.array(weights)
    s=np.mean(np.array(scores)* weights/weights.mean())
    if len(description):
        print(description, lim(s))
    return s



def get_model_params(model):
    weight_values = []
    for lay in model.layers:
        weight_values.extend( backend.batch_get_value(lay.weights))
    return weight_values



def set_model_params(model, weight_values):
    symb_weights = []
    for lay in model.layers:
        symb_weights.extend(lay.weights)
    assert len(symb_weights) == len(weight_values)
    for model_w, w in zip(symb_weights, weight_values):
        backend.set_value(model_w, w)
        
        
        
def save_model_weights(model, filename = 'fingerprint_model_weights.npz'):
    ws = get_model_params(model)
    np.savez(filename, ws)



def load_model_weights(model, filename = 'fingerprint_model_weights.npz'):
    ws = np.load(filename)
    set_model_params(model, ws[ws.keys()[0]])


def update_lr(model, initial_lr, relative_progress, total_lr_decay):
    """
    exponential decay
    
    initial_lr: any float (most reasonable values are in the range of 1e-5 to 1)
    total_lr_decay: value in (0, 1] -- this is the relative final LR at the end of training
    relative_progress: value in [0, 1] -- current position in training, where 0 == beginning, 1==end of training and a linear interpolation in-between
    """
    assert total_lr_decay > 0 and total_lr_decay <= 1
    backend.set_value(model.optimizer.lr, initial_lr * total_lr_decay**(relative_progress))
    
    


def train_model(model, train_data, valid_data, test_data, 
                 batchsize = 100, num_epochs = 100, train = True, 
                 initial_lr=3e-3, total_lr_decay=0.2, verbose = 1):
    """
    Main training loop for the DNN.
    
    Input:
    ---------    
    
    train_data, valid_data, test_data:
    
        lists of tuples (data-batch, labels-batch)
    
    total_lr_decay:
        
        value in (0, 1] -- this is the inverse total LR reduction factor over the course of training

    verbose:
        
        value in [0,1,2] -- 0 print minimal information (when training ends), 1 shows training loss, 2 shows training and validation loss after each epoch
    
    
    Returns:
    -----------
    
        model (keras model object) -- model/weights selected by early-stopping on the validation set (model at epoch with lowest validation error)
        
        3-tuple of train/validation/test dictionaries (as returned by eval_metrics_on(...) )
        
        2-tuple of training/validation-set MSE after each epoch of training
        
    """

    if train:
        
        log_train_mse = []
        log_validation_mse = []
        
        if verbose>0:
            print('starting training (compiling)...')
        
        best_valid = 9e9
        model_params_at_best_valid=[]
        
        times=[]
        for epoch in range(num_epochs):
            update_lr(model, initial_lr, epoch*1./num_epochs, total_lr_decay)
            batch_order = np.random.permutation(len(train_data))
            losses=[]
            rec_losses = []
            reg_losses = []
            pure_rec_losses = []
            t0 = time.clock()
            for i in batch_order:
                loss, rec_loss, reg_loss, pure_rec_loss = model.train_on_batch(x=train_data[i][0], y=[train_data[i][2], train_data[i][1]], check_batch_dim=False)
                losses.append(loss)
                rec_losses.append(rec_loss)
                reg_losses.append(reg_loss)
                pure_rec_losses.append(pure_rec_loss)
            times.append(time.clock()-t0)
            val_mse = test_on(valid_data,model,'valid_data score:' if verbose>1 else '')
            print(val_mse)
            if best_valid > val_mse:
                best_valid = val_mse
                model_params_at_best_valid = get_model_params(model) #kept in RAM (not saved to disk as that is slower)
            if verbose>0:
                print('Epoch',epoch+1,'completed with average reg loss',lim(np.mean(reg_losses)))
                print('Epoch',epoch+1,'completed with average pure rec loss',lim(np.mean(pure_rec_loss)))
                print('Epoch',epoch+1,'completed with average pure kl loss',lim(np.mean(rec_losses) - np.mean(pure_rec_loss)))
            log_train_mse.append(np.mean(losses))
            log_validation_mse.append(val_mse)
            
        # excludes times[0] as it includes compilation time
        print('Training @',lim(1./np.mean(times[1:])),'epochs/sec (',lim(batchsize*len(train_data)/np.mean(times[1:])),'examples/s)')
    
    
    #train_end  = test_on(train_data,model,'train mse (final):     ')
    #val_end    = test_on(valid_data,model,'validation mse (final):')
    #test_end   = test_on(test_data, model,'test  mse (final):     ')
    
    set_model_params(model, model_params_at_best_valid)
    train_labels = []
    for d in train_data:
        for x in d[2]:
            train_labels.append(x)
    train_labels = np.array(train_labels)
    val_labels = []
    for d in valid_data:
        val_labels.append(d[2])
    val_labels = np.array(val_labels)
    test_labels = []
    for d in test_data:
        test_labels.append(d[2])
    test_labels = np.array(test_labels)

    print(train_labels.shape)
    print(predict(train_data,model).shape)

    training_data_scores   = eval_metrics_on(predict(train_data,model), train_labels)
    validation_data_scores = eval_metrics_on(predict(valid_data,model), val_labels)
    test_predictions = predict(test_data,model)
    test_data_scores       = eval_metrics_on(test_predictions, test_labels)
    
    
    print('training set mse (best_val):  ', lim(training_data_scores['mse']))
    print('validation set mse (best_val):', lim(validation_data_scores['mse']))
    print('test set mse (best_val):      ', lim(test_data_scores['mse']))
    
    return model, (training_data_scores, validation_data_scores, test_data_scores), (log_train_mse, log_validation_mse), test_predictions






    
    
    
def crossvalidation_example(use_matrix_based_implementation = False):
    """
    Demonstration of data preprocessing, network configuration and (cross-validation) Training & testing
    
    There are two different (but equivalent!) implementations of neural-fingerprints, 
    which can be selected with the binary parameter <use_matrix_based_implementation>
    
    """
    # for reproducibility
    np.random.seed(1338)  
    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    num_epochs = 1
    batchsize  = 20   #batch size for training
    L2_reg     = 4e-3
    batch_normalization = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    fp_length = 51  # size of the final constructed fingerprint vector
    conv_width = 50 # number of filters per fingerprint layer
    fp_depth = 3    # number of convolutional fingerprint layers
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    n_hidden_units = 100
    predictor_MLP_layers = [100, 500]
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # total number of cross-validation splits to perform
    crossval_total_num_splits = 3#10
    
    
    # select the data that will be loaded or provide different data
    smilesarrays = []
    
    data, labels = utils.filter_data(utils.load_delaney)
    maxlength = -1
    chardict = {}
    count = 1
    for d in data:
        if len(d) > maxlength:
            maxlength = len(d)
        for i in range(len(d)):
            if d[i] not in chardict:
                chardict[d[i]] = count
                count = count + 1
    for d in data:
        coding = []
        for i in range(len(d)):
            coding.append(chardict[d[i]])
        coding = np.pad(coding, (0, maxlength - len(d)), 'constant', constant_values=(0, 0))
        temp = np.zeros((maxlength, len(chardict) + 1))
        temp[np.arange(maxlength), coding] = 1
        smilesarrays.append(temp)
    smilesarrays = np.array(smilesarrays)



#    data, labels = utils.filter_data(utils.load_Karthikeyan_MeltingPoints, data_cache_name='data/Karthikeyan_MeltingPoints')
    print('# of valid examples in data set:',len(data))
    

    
    train_mse = []
    val_mse   = []
    test_mse  = []
    test_scores = []
    all_test_predictions = []
    all_test_labels = [] #they have the original ordering of the data, but this might change if <cross_validation_split> changes
    
    if use_matrix_based_implementation:
        fn_build_model   = fingerprint_model_matrix_based.build_fingerprint_regression_model
    else:
        fn_build_model   = fingerprint_model_index_based.build_fingerprint_regression_model
    
    
    print('Naive baseline (using mean): MSE =', lim(np.mean((labels-labels.mean())**2)), '(RMSE =', lim(np.sqrt(np.mean((labels-labels.mean())**2))),')')
    
    
    
    for crossval_split_index in range(crossval_total_num_splits):
        print('\ncrossvalidation split',crossval_split_index+1,'of',crossval_total_num_splits)
        print(smilesarrays.shape)
        traindata, valdata, testdata = utils.cross_validation_split(data, smilesarrays.reshape((1127, 98*33)), labels, crossval_split_index=crossval_split_index,
                                                                    crossval_total_num_splits=crossval_total_num_splits, 
                                                                    validation_data_ratio=0.1)
        
        
        train, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(traindata, valdata, testdata,
                                                                     training_batchsize = batchsize, 
                                                                     testset_batchsize = 1000)
        print("B")
        print (np.array(train[0][1]).shape)
        print(labels.shape)
        model = fn_build_model(fp_length = fp_length, fp_depth = fp_depth, 
                               conv_width = conv_width, predictor_MLP_layers = predictor_MLP_layers, 
                               L2_reg = L2_reg, num_input_atom_features = 62, 
                               num_bond_features = 6, batch_normalization = batch_normalization)
        
        

        
        model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch, test_predictions = train_model(model, train, valid_data, test_data, 
                                     batchsize = batchsize, num_epochs = num_epochs, train=1)
        train_mse.append(train_scores_at_valbest['mse'])
        val_mse.append(val_scores_best['mse'])
        test_mse.append(test_scores_at_valbest['mse'])
        test_scores.append(test_scores_at_valbest)
        
        all_test_predictions.append(test_predictions[0])
        all_test_labels.append(np.concatenate(map(lambda x:x[-1],test_data)))
    


    
    print('\n\nCrossvalidation complete!\n')
    
    print('Mean training_data MSE =', lim(np.mean(train_mse)),                   '+-', lim(np.std(train_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean validation    MSE =', lim(np.mean(val_mse)),                     '+-', lim(np.std(val_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data     MSE =', lim(np.mean(test_mse)),                    '+-', lim(np.std(test_mse)/np.sqrt(crossval_total_num_splits)))
    print('Mean test_data RMSE    =', lim(np.mean(np.sqrt(np.array(test_mse)))), '+-', lim(np.std(np.sqrt(np.array(test_mse)))/np.sqrt(crossval_total_num_splits)))
    
    
    avg_test_scores = np.array([x.values() for x in test_scores]).mean(0)
    avg_test_scores_dict = dict(zip(test_scores[0].keys(), avg_test_scores))
    print()
    for k,v in avg_test_scores_dict.items():
        if k not in ['mse','rmse']:#filter duplicates
            print('Test-set',k,'=',lim(v))
                
    
    return model, test_data, chardict #this is the last trained model
    
    
    
    
if __name__=='__main__':
    
        
    # Two implementations are available (they are equivalent): index_based and matrix_based. 
    # The index_based one is usually slightly faster.    
    
    model, test_data, chardict = crossvalidation_example(use_matrix_based_implementation=1)
    chardict_rev = {}
    for key in chardict:
        chardict_rev[chardict[key]] = key
    chardict_rev[0] = ''
    
    model.save_weights('trained_model.txt')



    total_loss = 0
    for a in range(1):
        for b in range(20):
            print(b)
            model = regression_frozen()
            load_weights_by_name(model, 'trained_model.txt')

            ones_input = []
            test_labels = []
            for i in range(20):
                test_labels.append(test_data[0][2][0])
                ones_input.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            ones_input = np.array(ones_input)
            test_labels = np.array(test_labels)

            for i in range(0):
                model.train_on_batch(ones_input, test_labels)
            for layer in model.layers:
                if layer.name == 'getsample':
                    optmatrix = layer.get_weights()

            model = generate_smiles(optmatrix)
            load_weights_by_name(model, 'trained_model.txt')
            loss = model.train_on_batch(np.reshape(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), (1, 10)), np.reshape(np.array(test_data[0][1][0]), (1, 3234)))
            total_loss = total_loss + loss
    print(total_loss / 20)
    '''
    smiles_gen = model.predict(np.reshape(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), (1, 10)))
    smiles_gen = np.reshape(smiles_gen, (98, 33))
    actual = np.reshape(test_data[0][1][0], (98, 33))
    smiles = ""
    for whichchar in range(len(smiles_gen)):
        smiles += chardict_rev[np.argmax(smiles_gen[whichchar])]
    print(smiles)
    actualsmiles = ""
    for whichchar in range(len(actual)):
        actualsmiles += chardict_rev[np.argmax(actual[whichchar])]
    print(actualsmiles)
    mol = Chem.MolFromSmiles(smiles)
    print(_pyTPSA(mol) - test_data[0][1][0])
    '''





    #model.fit(zeros_input, train_labels)

    
    
    
    # to save the model weights use e.g.:
    #save_model_weights(model, 'trained_fingerprint_model.npz')
    
    # to load the saved model weights use e.g.:
    #load_model_weights(model, 'trained_fingerprint_model.npz')
    
    
    #this saves an image of the network's computational graph (an abstract form of it)
    # beware that this requires the 'graphviz' software!
    #save_model_visualization(model, filename = 'fingerprintmodel.png')




