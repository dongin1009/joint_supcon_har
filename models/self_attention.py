import tensorflow as tf

class AttentionWithContext(tf.keras.layers.Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.l
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False):
        super(AttentionWithContext, self).__init__()

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.u_constraint = tf.keras.constraints.get(u_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = tf.tensordot(x, self.W, axes=1)

        if self.bias:
            uit += self.b

        uit = tf.keras.activations.tanh(uit)
        ait = tf.tensordot(uit, self. u, axes=1)

        a = tf.math.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.cast(mask, tf.keras.backend.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= tf.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
                     tf.keras.backend.floatx())

        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        result = tf.keras.backend.sum(weighted_input, axis=1)

        if self.return_attention:
            return result, a
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value],
                                  [input_shape[0].value, input_shape[1].value])
        else:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
# def point_wise_feed_forward_network(d_model, dff):
#     return tf.keras.Sequential([
#         tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#         tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#     ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class SensorAttention(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(SensorAttention, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same', activation='relu')
        self.conv_f = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.ln(x)
        x1 = tf.expand_dims(x, axis=3)
        x1 = self.conv_1(x1)
        x1 = self.conv_f(x1)
        x1 = tf.keras.activations.softmax(x1, axis=2)
        x1 = tf.keras.layers.Reshape(x.shape[-2:])(x1)
        return tf.math.multiply(x, x1)


def create_model(n_timesteps, n_features, _dff=256, d_model=128, nh=4, dropout_rate=0.2, use_pe=False):
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features,))

    si = SensorAttention(n_filters=128, kernel_size=3, dilation_rate=2)(inputs)

    x = tf.keras.layers.Conv1D(d_model, 1, activation='relu')(si)

    if use_pe:
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        x = PositionalEncoding(n_timesteps, d_model)(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)    
    
    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
 
    # x = tf.keras.layers.Dense(n_outputs * 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    # predictions = tf.keras.layers.Dense(n_outputs, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="attention")

    return model


def Self_Attention_Encoder(input_shape, name="encoder"):
    inputs = tf.keras.layers.Input(input_shape)
    
    if input_shape[0] == 100: # pamap2:
        attention = create_model(100, 18, _dff=256, d_model=128, nh=4, dropout_rate=0.2, use_pe=False)
    elif input_shape[0] == 200: # wisdm:
        attention = create_model(200, 3, _dff=256, d_model=128, nh=4, dropout_rate=0.2, use_pe=False)
    output = attention(inputs)
    output = AttentionWithContext()(output)
    
    model = tf.keras.models.Model(inputs, output, name=name)
    return model