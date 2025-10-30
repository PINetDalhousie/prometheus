import tensorflow as tf
import pandas as pd

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Custom loss class for computing weighted categorical cross-entropy loss.
    
    This class inherits from `tf.keras.losses.Loss` and overrides the `call` method
    to compute the weighted categorical cross-entropy loss.
    
    Attributes:
        ratio: A float representing the weight for the minority class.
        name: A string representing the name of the loss.
    """

    def __init__(self, ratio, name='weighted_categorical_crossentropy'):
        """Initializes a new instance of the WeightedCategoricalCrossentropy class.
        
        Args:
            ratio: A float representing the weight for the minority class.
            name: A string representing the name of the loss. Defaults to 'weighted_categorical_crossentropy'.
        """
        super().__init__(name=name)
        self.ratio = ratio
        self.class_weight = tf.constant([[ratio, 1.0 - ratio]],dtype=tf.float32)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, y_true, y_pred):
        """Computes the weighted categorical cross-entropy loss.
        
        Args:
            y_true: A tensor of shape (batch_size, num_classes) representing the true labels.
            y_pred: A tensor of shape (batch_size, num_classes) representing the predicted probabilities.
            
        Returns:
            A scalar tensor representing the weighted categorical cross-entropy loss.
        """
        weights = tf.matmul(y_true, tf.transpose(self.class_weight))
        sample_losses = tf.reshape(self.categorical_crossentropy(y_true, y_pred),[-1,1])
        loss = tf.matmul(tf.transpose(sample_losses),weights)
        return loss
    

    @staticmethod
    def calculate_ratio(labels:pd.Series):
        """
        Calculates the failure to non failures ratio in labels.
        """
        return labels.value_counts()[1] / labels.value_counts()[0]                           


class TruncatedMSE(tf.keras.losses.Loss):
    """Custom loss class for computing truncated mean squared error loss.
    
    This class inherits from `tf.keras.losses.Loss` and overrides the `call` method
    to compute the truncated mean squared error loss.
    
    Attributes:
        name: A string representing the name of the loss.
    """

    def __init__(self, name='truncated_mse'):
        """Initializes a new instance of the TruncatedMSE class.
        
        Args:
            name: A string representing the name of the loss. Defaults to 'truncated_mse'.
        """
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """Computes the truncated mean squared error loss.
        
        Args:
            y_true: A tensor representing the true labels.
            y_pred: A tensor representing the predicted values.
            
        Returns:
            A tensor representing the truncated mean squared error loss.
        """
        # Truncate the label tensor to match the shape of the predicted tensor
        if y_pred.shape[1] < y_true.shape[1]:
            y_true = y_true[:, :y_pred.shape[1], :, :]

        # Compute the mean squared error loss
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        return mse_loss



if __name__ == '__main__':
    pass