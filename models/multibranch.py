import tensorflow as tf


def CNN_Set(input_shape, kernel_size, name="Set"):
    
    inputs = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=kernel_size, activation="relu")
    conv2 = tf.keras.layers.Conv1D(32, kernel_size=kernel_size, activation="relu")
    dropout = tf.keras.layers.Dropout(0.5)
    maxpooling = tf.keras.layers.MaxPool1D(pool_size=2)
    flatten = tf.keras.layers.Flatten()
    
    x = tf.keras.layers.TimeDistributed(conv1)(inputs)
    x = tf.keras.layers.TimeDistributed(conv2)(x)
    x = tf.keras.layers.TimeDistributed(dropout)(x)
    x = tf.keras.layers.TimeDistributed(maxpooling)(x)
    x = tf.keras.layers.TimeDistributed(flatten)(x)
    
    model = tf.keras.models.Model(inputs, x, name=name)
    return model

def Multibranch_Encoder(input_shape, name="encoder"):
    
    inputs = tf.keras.layers.Input(input_shape)
    if input_shape[0] == 100: # pamap2
        input_reshape = (4, 25, 18)
    elif input_shape[0] == 200: # wisdm
        input_reshape = (4, 50, 3)
    reshaped_inputs = tf.keras.layers.Reshape(input_reshape)(inputs)
    branch_1 = CNN_Set(input_reshape, kernel_size=3, name="branch_1")(reshaped_inputs)
    branch_2 = CNN_Set(input_reshape, kernel_size=7, name="branch_2")(reshaped_inputs)
    branch_3 = CNN_Set(input_reshape, kernel_size=11, name="branch_3")(reshaped_inputs)

    concat = tf.keras.layers.Concatenate()([branch_1, branch_2, branch_3])
    
    model = tf.keras.models.Model(inputs, concat, name=name)
    return model