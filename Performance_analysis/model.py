import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def buildModule(inputShape):
    drop_out_rate = 0.5
    inputLayer = layers.Input(shape = inputShape)
    
    x = layers.Conv1D(128,3, activation = 'relu')(inputLayer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(64,3, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(32,3, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.GlobalMaxPooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(inputLayer, x)
    # optimizer = tf.keras.optimizers.Adam()
    METRICS = [
      
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    optimizer = get_optimizer()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

    return model