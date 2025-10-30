'''
This class is responsible for managing different tensorflow metrics objects.
The main functionality is to keep track of multiple different types of metrics.
There are methods to initialize, reset and update metrics.
'''
import tensorflow as tf
from metrics import F1Score
import datetime
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler


class MetricsManager:
    """
    A class that manages multiple tensorflow metrics objects.

    Attributes:
        metrics (dict): A dictionary that stores metric objects with unique names.

    Methods:
        add_metric(name, metric): Adds a new metric to the dictionary with a given name.
        get_metric(name): Returns a metric object with a given name.
        reset_metrics(): Resets all metrics in the dictionary.
        update_metrics(y_true, y_pred): Updates all metrics in the dictionary with new predictions.
    """

    def __init__(self,current_time):
        """
        Initializes empty dictionaries to store train and validation metrics. Also, initializes 
        tensorboard summary writers to write metrics.
        """
        self.current_time = current_time
        self.train_metrics = {}
        self.validation_metrics = {}
        
        # initialize metrics
        self.initialize_train_metrics()
        self.initialize_validation_metrics()
        # Initialize tensorboard writer
        self.initialize_tensorboard_writer()


    def initialize_tensorboard_writer(self):
        train_log_dir = f'/users/grad/papry/MS_thesis-main/logs/{self.current_time}/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = f'/users/grad/papry/MS_thesis-main/logs/{self.current_time}/val'
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    def initialize_train_metrics(self):
        """
        Initializes precision,recall and f1score for individual classes
        """
        self.train_metrics.update({f"loss": tf.keras.metrics.Mean()})
        self.train_metrics.update({f"precision_0": tf.keras.metrics.Precision(class_id=0)})
        self.train_metrics.update({f"recall_0": tf.keras.metrics.Recall(class_id=0)})
        self.train_metrics.update({f"precision_1": tf.keras.metrics.Precision(class_id=1)})
        self.train_metrics.update({f"recall_1": tf.keras.metrics.Recall(class_id=1)})

    def initialize_validation_metrics(self):
        """
        Initializes precision,recall and f1score for individual classes
        """
        self.validation_metrics.update({f"loss": tf.keras.metrics.Mean()})
        self.validation_metrics.update({f"precision_0": tf.keras.metrics.Precision(class_id=0)})
        self.validation_metrics.update({f"recall_0": tf.keras.metrics.Recall(class_id=0)})
        self.validation_metrics.update({f"precision_1": tf.keras.metrics.Precision(class_id=1)})
        self.validation_metrics.update({f"recall_1": tf.keras.metrics.Recall(class_id=1)})


    def reset_metrics(self):
        """
        Resets all metrics in the dictionary.

        Args:
            None

        Returns:
            None
        """
        for metric in self.train_metrics.values():
            metric.reset_states()
        for metric in self.validation_metrics.values():
            metric.reset_states()


    def update_train_meteric(self, loss, labels, predictions):
        """
        Updates all train metrics in the dictionary

        Args:
            labels (tf.Tensor): The true labels for the predictions.
            predictions (tf.Tensor): The predicted labels.

        Returns:
            None
        """
        for metric_key, metric in self.train_metrics.items():
            if metric_key == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, predictions)
    
    
    def update_validation_meteric(self,loss,labels,predictions):
        """
        Updates all validation metrics in the dictionary

        Args:
            labels (tf.Tensor): The true labels for the predictions.
            predictions (tf.Tensor): The predicted labels.

        Returns:
            None
        """
        for metric_key, metric in self.validation_metrics.items():
            if metric_key == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, predictions)



    @staticmethod
    def calculate_f1score(precision,recall):
        return (2 * precision * recall) / (precision + recall + 1e-7)


    def write_metrics(self,epoch):
        """
        Writes the training and validation metrics to the summary writers.

        Arguments:
            epoch: The current epoch number.
        """

        # Write training metrics to the train summary writer
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_metrics["loss"].result(), step=epoch)
            tf.summary.scalar('precision_0', self.train_metrics["precision_0"].result(),step=epoch)
            tf.summary.scalar('recall_0', self.train_metrics["recall_0"].result(),step=epoch)
            tf.summary.scalar('f1score_0', MetricsManager.calculate_f1score(self.train_metrics["precision_0"].result(),
                                                                            self.train_metrics["recall_0"].result()),step=epoch)
            tf.summary.scalar('precision_1', self.train_metrics["precision_1"].result(),step=epoch)
            tf.summary.scalar('recall_1', self.train_metrics["recall_1"].result(),step=epoch)
            tf.summary.scalar('f1score_1', MetricsManager.calculate_f1score(self.train_metrics["precision_1"].result(),
                                                                            self.train_metrics["recall_1"].result()),step=epoch)

        # Write validation metrics to the validation summary writer
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.validation_metrics["loss"].result(), step=epoch)
            tf.summary.scalar('precision_0', self.validation_metrics["precision_0"].result(),step=epoch)
            tf.summary.scalar('recall_0', self.validation_metrics["recall_0"].result(),step=epoch)
            tf.summary.scalar('f1score_0', MetricsManager.calculate_f1score(self.validation_metrics["precision_0"].result(),
                                                                            self.validation_metrics["recall_0"].result()),step=epoch)
            tf.summary.scalar('precision_1', self.validation_metrics["precision_1"].result(),step=epoch)
            tf.summary.scalar('recall_1', self.validation_metrics["recall_1"].result(),step=epoch)
            tf.summary.scalar('f1score_1', MetricsManager.calculate_f1score(self.validation_metrics["precision_1"].result(),
                                                                            self.validation_metrics["recall_1"].result()),step=epoch)


class TestMetricsManager:
    def __init__(self,**kwargs):
        # define the precision and recall metrics for class 0/non failure
        self.precision_0_test = tf.keras.metrics.Precision(class_id=0)
        self.recall_0_test = tf.keras.metrics.Recall(class_id=0)
        # define the precision and recall metrics for class 1/failure
        self.precision_1_test = tf.keras.metrics.Precision(class_id=1)
        self.recall_1_test = tf.keras.metrics.Recall(class_id=1)
        # define the threshold for autoencoder
        if "threshold" in kwargs:
            self.threshold = kwargs["threshold"]
        self.reconstruction_errors = []
        self.labels = []
        self.approach = kwargs["approach"]

    def update_test_metrics(self,labels,predictions):
        self.precision_0_test.update_state(labels, predictions)
        self.recall_0_test.update_state(labels, predictions)
        self.precision_1_test.update_state(labels, predictions)
        self.recall_1_test.update_state(labels, predictions)

    def append_errors_labels(self,y_pred,kpis,y_true):
        # calculate the reconstruction error
        difference = tf.math.squared_difference(kpis, y_pred).numpy()
        #difference = tf.math.abs(kpis-y_pred).numpy()

        # sum the reconstruction error
        if self.approach == "prev":
            difference = tf.reduce_sum(difference, axis=[1,2]).numpy()
        elif self.approach == "new":
            difference = tf.reduce_sum(difference, axis=[1,2,3]).numpy()
        
        # append the reconstruction errors to a list
        self.reconstruction_errors.append(difference)
        self.labels.append(y_true.numpy()[:,1])
        
        return difference

    def calculate_binary_predictions(self,y_pred,kpis,y_true):
        # calculate the reconstruction error
        difference = self.append_errors_labels(y_pred,kpis,y_true)
        # calculate the binary predictions
        binary_predictions = tf.where(difference > self.threshold, 1, 0)
        # one hot encode the binary predictions
        binary_predictions = tf.one_hot(binary_predictions, depth=2)
        return binary_predictions

    def update_autoencoder_test_metrics(self,y_true,y_pred,kpis):
        if self.threshold != None:
            # calculate the binary predictions
            y_pred = self.calculate_binary_predictions(y_pred,kpis,y_true)
        else:
            # calculate the reconstruction error
            _ = self.append_errors_labels(y_pred,kpis,y_true)

        # update the metrics
        self.precision_0_test.update_state(y_true, y_pred)
        self.recall_0_test.update_state(y_true, y_pred)
        self.precision_1_test.update_state(y_true, y_pred)
        self.recall_1_test.update_state(y_true, y_pred)

class OptimalThresholdCalculator:

    def __init__(self,test_metrics:TestMetricsManager):
        self.test_metrics = test_metrics
        self.errors = self.test_metrics.reconstruction_errors
        self.labels = self.test_metrics.labels

        # prepare the test metrics for threshold calculation
        self.prepare_test_metrics_for_threshold_calculation()

    def prepare_test_metrics_for_threshold_calculation(self):
        # Concatenate the arrays into a single NumPy array
        self.errors = np.concatenate(self.errors)
        self.labels = np.concatenate(self.labels)
        
        # Convert the NumPy array to a list
        self.errors = self.errors.tolist()
        self.labels = self.labels.tolist()

    def __sort(self):
        # Use zip to combine the two lists into tuples
        combined = zip(self.errors, self.labels)

        # Sort the combined list based on the values in list1
        sorted_combined = sorted(combined, key=lambda x: x[0])

        # Unzip the sorted combined list back into separate lists
        self.errors, self.labels = zip(*sorted_combined)

    def calculate_threshold_v0(self):
        # Sort reconstruction errors in ascending order
        self.__sort()

        # check last 200 values
        # print(self.errors[-200:])
        # print(self.labels[-200:])
        # print(asd)

        # Initialize lists for storing F1 scores and accuracies
        f1_scores = []
        num_instances = len(self.labels)

        counter = 0
        for threshold in self.errors:
            # Classify instances based on the reconstruction error and threshold
            predicted_labels = [1 if error > threshold else 0 for error in self.errors]
            macro_f1_score = f1_score(self.labels, predicted_labels, average='macro')
            f1_scores.append(macro_f1_score*100)
            counter += 1
            print(f"done with {counter} out of {num_instances}; f1_score: {macro_f1_score*100}")

        # Find the optimal threshold
        optimal_threshold = self.errors[f1_scores.index(max(f1_scores))]
        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Max F1 Score: {max(f1_scores)}")
        return optimal_threshold
    

    def calculate_threshold_v1(self):
        
        self.errors = np.array(self.errors)
        # Sort reconstruction errors in ascending order
        sorted_indices = np.argsort(self.errors)
        sorted_errors = self.errors[sorted_indices]
        sorted_labels = np.array(self.labels)[sorted_indices]

        f1_scores = []
        num_instances = len(self.labels)
        max_f1_score = 0
        optimal_threshold = None

        for idx in range(num_instances - 1, 0, -1):
            # Calculate F1 score for the current threshold
            predicted_labels = np.where(sorted_errors > sorted_errors[idx], 1, 0)
            macro_f1_score = f1_score(sorted_labels, predicted_labels, average='macro')
            f1_scores.append(macro_f1_score * 100)

            if macro_f1_score > max_f1_score:
                max_f1_score = macro_f1_score * 100
                optimal_threshold = sorted_errors[idx]

        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Max Macro F1 Score: {max_f1_score}")
        return optimal_threshold
    
    def calculate_threshold(self):
        
        self.errors = np.array(self.errors)
        self.labels = np.array(self.labels)

        # Minmax scale the reconstruction errors
        scalar = MinMaxScaler()
        self.errors = scalar.fit_transform(self.errors.reshape(-1, 1)).flatten()

        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(self.labels,self.errors)

        # Calculate F1-score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)

        # Find the threshold that maximizes F1-score
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Rescale optimal threshold to original scale
        optimal_threshold = scalar.inverse_transform([[optimal_threshold]])[0][0]

        print(f"Optimal Threshold: {optimal_threshold}")
        return optimal_threshold

        

if __name__ == '__main__':
    pass