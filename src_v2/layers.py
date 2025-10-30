import tensorflow as tf
import numpy as np



class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, num_features, dropout=0, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_features = num_features

        # Attention Part
        self.layer_norm_1 = tf.keras.layers.BatchNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )
        self.dropout_layer_1 = tf.keras.layers.Dropout(self.dropout)

        # Feed Forward Part
        self.layer_norm_2 = tf.keras.layers.BatchNormalization()
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.dropout_layer_2 = tf.keras.layers.Dropout(self.dropout)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=self.num_features, kernel_size=1)

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        attention_output = self.attention(x, x)
        x = self.dropout_layer_1(attention_output)
        res = x + inputs

        x = self.layer_norm_2(res)
        x = self.conv1d_1(x)
        x = self.dropout_layer_2(x)
        x = self.conv1d_2(x)
        return x + res


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(
        **kwargs
    )
    self.layernorm = tf.keras.layers.LayerNormalization()
    #self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    x = self.mha(
        query=x,
        value=x,
        key=x)
    #x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.0):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    seq_output = self.seq(x)
    x = self.add([x, seq_output])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.0):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        value_dim=d_model,
        output_shape=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, timesteps,**kwargs):
        """
        Initializes an ExpandDims layer.

        Args:
            axis (int): The axis along which to expand the input tensor.
        """
        super(ExpandDims, self).__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, axis=1), repeats=self.timesteps, axis=1)

class SeperateCatFeatures(tf.keras.layers.Layer):
    def __init__(self, cat_features_number,**kwargs):
        super(SeperateCatFeatures, self).__init__(**kwargs)
        self.cat_features_number = cat_features_number

    def call(self, inputs):
        temporal_features = inputs[:,:,:-self.cat_features_number]
        static_features = inputs[:,0,-self.cat_features_number:]
        return temporal_features,static_features
    


class MaxReduction(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        """
        Initializes a MaxReduction layer.

        Args:
            axis (int): The axis along which to reduce the input tensor.
        """
        super(MaxReduction, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis)
    
class SumReduction(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        """
        Initializes a MaxReduction layer.

        Args:
            axis (int): The axis along which to reduce the input tensor.
        """
        super(SumReduction, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
    

class ReshapeBatch(tf.keras.layers.Layer):
    def __init__(self, prev_days_data,num_features,**kwargs):
        """
        Initializes a ReshapeBatch layer.

        Args:
            prev_days_data (int): The number of previous days of data.
            num_features (int): The number of features in each data point.
        """
        super(ReshapeBatch, self).__init__(**kwargs)
        self.prev_days_data = prev_days_data
        self.num_features = num_features

    def call(self, inputs):
        return tf.reshape(inputs,[-1,self.prev_days_data,self.num_features])
    
class ReshapeBatchWS(tf.keras.layers.Layer):
    def __init__(self,num_features,**kwargs):
        """
        Reshape to meet weather station repeat format
        """
        super(ReshapeBatchWS, self).__init__(**kwargs)
        self.num_features = num_features

    def call(self, inputs):
        return tf.reshape(inputs,[-1,self.num_features])
    
class ReshapeBatchInverse(tf.keras.layers.Layer):
    def __init__(self, prev_days_data,num_features,batch_size,**kwargs):
        """
        Initializes a ReshapeBatchInverse layer.

        Args:
            prev_days_data (int): The number of previous days of data.
            num_features (int): The number of features in each data point.
            batch_size (int): Batch size.
        """
        super(ReshapeBatchInverse, self).__init__(**kwargs)
        self.prev_days_data = prev_days_data
        self.num_features = num_features
        self.batch_size = batch_size
    
    def call(self, inputs):
        return tf.reshape(inputs,[self.batch_size,-1,self.prev_days_data,self.num_features])
    
class ReshapeAndSliceStaticFeatures(tf.keras.layers.Layer):
    def __init__(self, batch_size,cat_features_number,**kwargs):
        super(ReshapeAndSliceStaticFeatures, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.cat_features_number = cat_features_number

    def call(self, inputs):
        reshaped =  tf.reshape(inputs,[self.batch_size,-1,self.cat_features_number])
        return reshaped[:,0,:]
    

class ReshapeLSTMOutput(tf.keras.layers.Layer):
    def __init__(self, batch_size, lstm_output,**kwargs):
        super(ReshapeLSTMOutput, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.lstm_output = lstm_output

    def call(self, inputs):
        return tf.reshape(inputs,[self.batch_size,-1,self.lstm_output])
    

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding layer for adding positional information to input sequences.
    """

    def __init__(self, position, d_model):
        """
        Initialize the PositionalEncoding layer.

        Args:
            position (int): Maximum position in the sequence.
            d_model (int): Dimensionality of the model.
        """
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.compute_positional_encoding(position, d_model)

    def compute_positional_encoding(self, position, d_model):
        """
        Compute the positional encoding matrix.

        Args:
            position (int): Maximum position in the sequence.
            d_model (int): Dimensionality of the model.

        Returns:
            A positional encoding matrix of shape (position, d_model).
        """
        print("d_model in PE ", d_model),
        encoding = np.zeros((position, d_model))
        for pos in range(position):
            for i in range(d_model):
                encoding[pos, i] = pos / np.power(10000, (2 * (i // 2)) / d_model)
                if i % 2 == 1:
                    encoding[pos, i] = np.sin(encoding[pos, i])
                else:
                    encoding[pos, i] = np.cos(encoding[pos, i])
        return tf.cast(encoding, dtype=tf.float32)

    def call(self, inputs):
        """
        Apply positional encoding to the inputs.

        Args:
            inputs: Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            A tensor with positional encoding added to the inputs.
        """
        shape = tf.shape(inputs)
        batch_size, seq_length = shape[0], shape[1]
        position_encoding = tf.repeat(tf.expand_dims(self.positional_encoding[:seq_length, :], axis=0), batch_size, axis=0)
        return inputs + position_encoding
    
if __name__ == "__main__":
    pass