import tensorflow as tf
from metrics_manager import MetricsManager

class TrainVal:
    
    def __init__(self,model:tf.keras.Model,loss_object,optimizer) -> None:
        '''
        Attributes:
            model: a tf.keras.Model
            loss_object: a tensorflow loss object
            optimizer: optimizer to use
        '''
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer

    @tf.function
    def train_step(self,kpis,labels):
        with tf.GradientTape() as tape:
            predictions = self.model(kpis, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, predictions

    @tf.function
    def validation_step(self,kpis,labels):
        predictions = self.model(kpis, training=False)
        loss = self.loss_object(labels, predictions)
        return loss, predictions
    
    





if __name__ == '__main__':
    pass