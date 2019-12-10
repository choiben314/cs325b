import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from modules.run import Runner

class Metrics(Callback):
    
    def __init__(self, val_data, tensorboard_dir, val_labels, n_epochs):
        super(Metrics, self).__init__()
        self.validation_data = val_data
        self.logdir = tensorboard_dir
        self.file_writer = tf.summary.create_file_writer(self.logdir + '/validation/metrics')
        self.file_writer.set_as_default()
        self.val_labels = val_labels
        self.num_epochs = n_epochs
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = None
        val_targ = None
        
        if self.val_labels is not None:
            val_predict = np.argmax(self.model.predict(self.validation_data, steps=self.num_epochs), axis=1)
            val_targ = self.val_labels
        else:
            val_predict = np.argmax(self.model.predict(self.validation_data), axis=1)
            val_targ = self.validation_data.labels

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        _val_f1_weighted = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        
        tf.summary.scalar('f1_macro', data=_val_f1, step=epoch)
        tf.summary.scalar('f1_micro', data=_val_f1_micro, step=epoch)
        tf.summary.scalar('f1_weighted', data=_val_f1_weighted, step=epoch)
        self.file_writer.flush()
        
        return _val_f1

class Trainer(Runner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        checkpoints = True
        tensorboard = True
        
        self.init_directories(checkpoints, tensorboard)
        
        self.init_callbacks(checkpoints, tensorboard)
        
        self.init_optimizer()
        
        self.validation_data = args[-1]
        
    def init_callbacks(self, checkpoints=True, tensorboard=True):
        self.tensorboard_callback = TensorBoard(
            self.tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq=self.config["batch_size"] * self.config["tensorboard_freq"]
        )

        self.checkpoints_callback = ModelCheckpoint(
            os.path.join(self.checkpoints_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch',
        )
        
#         self.metrics_callback = Metrics(self.validation_data)
        
        self.callbacks = [self.tensorboard_callback, self.checkpoints_callback] #self.metrics_callback]
    
    def init_optimizer(self):
        if self.config["optimizer"] == "sgd":
            self.optimizer = SGD(learning_rate=self.config["learning_rate"])
        elif self.config["optimizer"] == "adam":
            self.optimizer = Adam(learning_rate=self.config["learning_rate"])
        else:
            raise ValueError