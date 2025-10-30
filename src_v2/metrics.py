import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class F1Score_OLD(tf.keras.metrics.Metric):
    def __init__(self, name, class_id, **kwargs):
        super(F1Score, self).__init__(name=name,**kwargs)
        self.precision = tf.keras.metrics.Precision(class_id)
        self.recall = tf.keras.metrics.Recall(class_id)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return ((2 * precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, class_id, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight(name='true_positives', initializer='zeros')
        self.fp = self.add_weight(name='false_positives', initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)), tf.float32), axis=None)
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)), tf.float32), axis=None)
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, self.class_id), tf.not_equal(y_pred, self.class_id)), tf.float32), axis=None)
        self.tp.assign_add(true_positives)
        self.fp.assign_add(false_positives)
        self.fn.assign_add(false_negatives)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1_score

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class PercentileMAE(tf.keras.metrics.Metric):
    def __init__(self, name='percentile_mae_diff', **kwargs):
        super(PercentileMAE, self).__init__(name=name, **kwargs)
        self.percentiles = self.add_weight(name='percentiles', initializer='zeros')
        self.counts = self.add_weight(name='counts', initializer='zeros')
        self.percentile = 99

    def update_state(self, y_true, y_pred, sample_weight=None):
        errors = tf.abs(y_true - y_pred)
        # Reduce the tensor along the second and third axes
        errors = tf.reduce_mean(errors, axis=[1, 2])
        percentile_value = tfp.stats.percentile(errors, self.percentile)
        self.percentiles.assign_add(percentile_value)
        self.counts.assign_add(1)

    def result(self):
        return self.percentiles/self.counts

    def reset_state(self):
        self.percentiles.assign(0.0)
        self.counts.assign(0)



class ReconstructionError(tf.keras.metrics.Metric):
    """
    Custom TensorFlow Keras metric to calculate the reconstruction error.
    
    The reconstruction error is computed as the squared difference between the true values (y_true) and
    the predicted values (y_pred). It is then summed along specified axes and averaged across all samples.
    
    Args:
        name: (Optional) String name of the metric.
    """

    def __init__(self, name='reconstruction_error', **kwargs):
        super(ReconstructionError, self).__init__(name=name, **kwargs)
        self.reconstruction_error = self.add_weight(name='total_reconstruction_error', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by calculating the reconstruction error.

        Args:
            y_true: The true values.
            y_pred: The predicted values.
            sample_weight: (Optional) Optional weighting of individual samples (not used in this metric).
        """
        # Calculate the reconstruction error
        difference = tf.math.abs(y_true - y_pred)
        reconstruction_error = tf.reduce_sum(difference, axis=[1, 2])

        # Update the total reconstruction error and total samples
        self.reconstruction_error.assign_add(tf.reduce_sum(reconstruction_error))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        """
        Computes the average reconstruction error.

        Returns:
            The average reconstruction error as a floating-point tensor.
        """
        # Calculate the average reconstruction error
        avg_reconstruction_error = self.reconstruction_error / self.total_samples
        return avg_reconstruction_error

    def reset_state(self):
        """
        Resets the state of the metric.

        This method is called at the start of each epoch to reset the total reconstruction error
        and total samples count.
        """
        self.reconstruction_error.assign(0.0)
        self.total_samples.assign(0.0)


class TruncateReconstructionError(tf.keras.metrics.Metric):
    """
    Custom TensorFlow Keras metric to calculate the reconstruction error.
    
    The reconstruction error is computed as the squared difference between the true values (y_true) and
    the predicted values (y_pred). It is then summed along specified axes and averaged across all samples.
    
    Args:
        name: (Optional) String name of the metric.
    """

    def __init__(self, name='reconstruction_error', **kwargs):
        super(TruncateReconstructionError, self).__init__(name=name, **kwargs)
        self.reconstruction_error = self.add_weight(name='total_reconstruction_error', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by calculating the reconstruction error.

        Args:
            y_true: The true values.
            y_pred: The predicted values.
            sample_weight: (Optional) Optional weighting of individual samples (not used in this metric).
        """
        # Truncate the label tensor to match the shape of the predicted tensor
        if y_pred.shape[1] < y_true.shape[1]:
            y_true = y_true[:, :y_pred.shape[1], :, :]

        # Calculate the reconstruction error
        difference = tf.math.abs(y_true - y_pred)
        reconstruction_error = tf.reduce_sum(difference, axis=[1, 2])

        # Update the total reconstruction error and total samples
        self.reconstruction_error.assign_add(tf.reduce_sum(reconstruction_error))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        """
        Computes the average reconstruction error.

        Returns:
            The average reconstruction error as a floating-point tensor.
        """
        # Calculate the average reconstruction error
        avg_reconstruction_error = self.reconstruction_error / self.total_samples
        return avg_reconstruction_error

    def reset_state(self):
        """
        Resets the state of the metric.

        This method is called at the start of each epoch to reset the total reconstruction error
        and total samples count.
        """
        self.reconstruction_error.assign(0.0)
        self.total_samples.assign(0.0)


if __name__ == '__main__':
    pass