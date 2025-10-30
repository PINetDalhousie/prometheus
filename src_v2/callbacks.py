import tensorflow as tf
import numpy as np

class Callbacks:

    def __init__(
        self, 
        current_time, 
        **kwargs
    ):
        
        self.callbacks = {
            'tensorboard': tf.keras.callbacks.TensorBoard(
                f'/home/papry/Exp_Project/MS_thesis-main/logs/{current_time}',
                write_graph=True,
                update_freq="epoch",
                profile_batch=(1,20)
                ),
            'csv_logger': tf.keras.callbacks.CSVLogger(
                f'/home/papry/Exp_Project/MS_thesis-main/logs/{current_time}/training.log'
                ),
            'model_checkpoint': tf.keras.callbacks.ModelCheckpoint(
                filepath=f'/home/papry/Exp_Project/MS_thesis-main/logs/{current_time}/ckpt',
                save_weights_only=False,
                monitor='val_f1score_1',
                verbose=1,
                mode='max',
                save_best_only=True
                ),
            #'early_stopping': tf.keras.callbacks.EarlyStopping(
              #  monitor='val_f1score_1',
               # patience=200,
              #  verbose=1,
              #  factor=0.1,
              #  restore_best_weights=True,
               # min_lr=0.00001,
              #  mode='max'
              #  )
            
            }
    
    def add_callback(self,name,callback):
        
        self.callbacks[name] = callback
    

    def remove_callback(self,name):    
        self.callbacks.pop(name)

    def get_callbacks(
        self
    ):
        return list(self.callbacks.values())
    

class ResetPercentileCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, track_metric='val_mae'):
        super(ResetPercentileCallback, self).__init__()
        self.lowest_loss = float('inf')
        self.best_percentile = None
        self.track_metric = track_metric
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.track_metric)
        current_percentile = logs.get('val_pae')

        if current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_percentile = current_percentile

        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar('Best Percentile', self.best_percentile, step=epoch)



class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, monitor, save_best_only, mode='auto'):
        super(SaveModelCallback, self).__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_metric_value = np.Inf if mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_metric_value = logs.get(self.monitor)
        if current_metric_value is None:
            return

        if self.mode == 'min':
            improvement = current_metric_value < self.best_metric_value
        else:
            improvement = current_metric_value > self.best_metric_value

        if improvement:
            self.best_metric_value = current_metric_value
            if self.save_best_only:
                self.model.save(filepath=self.save_dir)
                print(f"Best model saved at epoch {epoch+1}.")
        elif not self.save_best_only:
            self.model.save(filepath=self.save_dir.format(epoch+1))
            print(f"Model saved at epoch {epoch+1}.")

if __name__ == "__main__":
    pass