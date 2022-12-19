import tensorflow as tf

from models.deepconvlstm import DeepConvLSTM_Encoder
from models.self_attention import Self_Attention_Encoder
from models.multibranch import Multibranch_Encoder

def create_encoder(encoder_name, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    if encoder_name == 'deepconvlstm':
        encoder = DeepConvLSTM_Encoder(input_shape)(inputs)
        
    elif encoder_name == 'self_attention':
        encoder = Self_Attention_Encoder(input_shape)(inputs)
    
    elif encoder_name == 'multibranch':
        encoder = Multibranch_Encoder(input_shape)(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=encoder, name=encoder_name+"_encoder")
    return model

def create_classifier(encoder_name, encoder, input_shape, num_class, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    if encoder_name == 'self_attention':
        features = tf.keras.layers.Dense(num_class*4, activation='relu')(features)
        features = tf.keras.layers.Dropout(0.2)(features)
        
    elif encoder_name == 'multibranch':
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(features)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False))(x)
        x = tf.keras.layers.Dense(128)(x)
        features = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(num_class, activation='softmax', name="classified")(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    return model

def add_contrastive_head(encoder_name, encoder, input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    if encoder_name == 'multibranch':
        features = tf.keras.layers.GlobalAveragePooling1D()(features)
    x = tf.keras.layers.LayerNormalization()(features)
    nonactivated = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Activation("relu", name="contrastive")(nonactivated)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder_with_projection-head")
    return model

def joint_supcon(encoder_name, encoder, input_shape, num_class, contrastive_shape):
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    cls_features, con_features = features, features
    if encoder_name == 'self_attention':
        cls_features = tf.keras.layers.Dense(num_class*4, activation='relu')(cls_features)
        cls_features = tf.keras.layers.Dropout(0.2)(cls_features)
        
    elif encoder_name == 'multibranch':
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(cls_features)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False))(x)
        x = tf.keras.layers.Dense(128)(x)
        cls_features = tf.keras.layers.BatchNormalization()(x)
    
    cls_features = tf.keras.layers.Dense(num_class, activation='softmax', name="classified")(cls_features)
    
    if encoder_name == 'multibranch':
        con_features = tf.keras.layers.GlobalAveragePooling1D()(con_features)
    x = tf.keras.layers.LayerNormalization()(con_features)
    nonactivated = tf.keras.layers.Dense(contrastive_shape)(x)
    con_features = tf.keras.layers.Activation("relu", name="contrastive")(nonactivated)
    
    model = tf.keras.Model(inputs=inputs, outputs=[cls_features, con_features], name="joint-supcon")
    return model