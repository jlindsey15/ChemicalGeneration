
import keras.optimizers as optimizers
import keras.regularizers as regularizers
import keras.models as models
import keras.layers as layers
import keras.backend as backend
import keras.objectives
from keras.engine.topology import Merge
from keras.constraints import maxnorm

degrees = range(1,5)



def neural_fingerprint_layer(inputs, atom_features_of_previous_layer, num_atom_features, 
                             conv_width, fp_length, L2_reg, num_bond_features , 
                             batch_normalization = False, layer_index=0):
    '''
    one layer of the "convolutional" neural-fingerprint network
    
    This implementation uses indexing to select the features of neighboring atoms, and binary matrices to map atoms in the batch to the indiviual molecules in the batch.
    '''
#    atom_features_of_previous_layer has shape: (variable_a, num_input_atom_features) [if first layer] or (variable_a, conv_width)
    
    activations_by_degree = []
    
    
    for degree in degrees:
        
        atom_features_of_previous_layer_this_degree = layers.Lambda(lambda x: backend.dot(inputs['atom_features_selector_matrix_degree_'+str(degree)], x))(atom_features_of_previous_layer) # layers.Lambda(lambda x: backend.dot(inputs['atom_features_selector_matrix_degree_'+str(degree)], x))(atom_features_of_previous_layer)
        

        merged_atom_bond_features = layers.merge([atom_features_of_previous_layer_this_degree, inputs['bond_features_degree_'+str(degree)]], mode='concat', concat_axis=1)

        activations = layers.Dense(conv_width, activation='relu', bias=False, name='activations_{}_degree_{}'.format(layer_index, degree))(merged_atom_bond_features)

        activations_by_degree.append(activations)

    # skip-connection to output/final fingerprint
    output_to_fingerprint_tmp = layers.Dense(fp_length, activation='softmax', name = 'fingerprint_skip_connection_{}'.format(layer_index))(atom_features_of_previous_layer) # (variable_a, fp_length)
    #(variable_a, fp_length)
    output_to_fingerprint     = layers.Lambda(lambda x: backend.dot(inputs['atom_batch_matching_matrix_degree_'+str(degree)], x))(output_to_fingerprint_tmp)  # layers.Lambda(lambda x: backend.dot(inputs['atom_batch_matching_matrix_degree_'+str(degree)], x))(output_to_fingerprint_tmp) # (batch_size, fp_length)

    # connect to next layer
    this_activations_tmp = layers.Dense(conv_width, activation='relu', name='layer_{}_activations'.format(layer_index))(atom_features_of_previous_layer) # (variable_a, conv_width)
    # (variable_a, conv_width)
    merged_neighbor_activations = layers.merge(activations_by_degree, mode='concat',concat_axis=0)

    new_atom_features = layers.Lambda(lambda x:merged_neighbor_activations + x)(this_activations_tmp ) #(variable_a, conv_width)
    if batch_normalization:
        new_atom_features = layers.normalization.BatchNormalization()(new_atom_features)

    #new_atom_features = layers.Lambda(backend.relu)(new_atom_features) #(variable_a, conv_width)
    
    return new_atom_features, output_to_fingerprint









def regression_frozen():
    """
    fp_length   # Usually neural fps need far fewer dimensions than morgan.
    fp_depth     # The depth of the network equals the fingerprint radius.
    conv_width   # Only the neural fps need this parameter.
    h1_size     # Size of hidden layer of network on top of fps.
    
    """
    
    zeros = layers.Input(name='properties', shape=(10,))
    sample = layers.Dense(10, activation='linear', name='getsample', b_constraint=maxnorm(0))(zeros)
    
    
    layer = layers.Dense(100, activation='linear', name='regression')
    layer.trainable = False
    regression = layer(sample)
    layer = layers.Dense(1, activation='relu', name='regression2')
    layer.trainable=False
    regression = layer(regression)

    model = models.Model(input=zeros, output=[regression])
    model.compile(optimizer=optimizers.Adam(), loss={'regression2':'mse'})
    return model






def generate_smiles(getsampleweights, predictor_MLP_layers = [100, 500], L2_reg=4e-3):
    """
        fp_length   # Usually neural fps need far fewer dimensions than morgan.
        fp_depth     # The depth of the network equals the fingerprint radius.
        conv_width   # Only the neural fps need this parameter.
        h1_size     # Size of hidden layer of network on top of fps.
        
        """
    
    ones = layers.Input(name='properties', shape=(10,))
    layer = layers.Dense(10, weights=getsampleweights, activation='linear', name='getsample')
    layer.trainable = False
    sample = layer(ones)
    
    
    Prediction_MLP_layer = sample


    for i, hidden in enumerate(predictor_MLP_layers):
        
        layer = layers.Dense(hidden, activation='relu', W_regularizer=regularizers.l2(L2_reg), name='MLP_hidden_'+str(i))
        layer.trainable = False
        Prediction_MLP_layer = layer(Prediction_MLP_layer)



    layer = layers.Dense(98*33, activation='sigmoid', name='reconstruction')
    layer.trainable = False
    reconstruction = layer(Prediction_MLP_layer)

    
    model = models.Model(input=ones, output=[reconstruction])
    model.compile(optimizer=optimizers.Adam(), loss={'reconstruction':'binary_crossentropy'})
    return model



