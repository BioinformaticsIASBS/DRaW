# !pip install tensorflow-addons
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc


rootDir = 'covid/'
rootDirSave = 'covid/models'
rootDirSaveReper = 'covid/drugReper/'
# rootDir = ''
files = [
    'drugsim.csv',
    'virusdrug.csv',
    'virussim.csv',
]

drugVirus = pd.read_csv(rootDir + files[1], delimiter = ',', header=None, encoding='cp1252').to_numpy()
drugNames = drugVirus[1:, 0]
virusNames = drugVirus[0, 1:]
drugVirus = drugVirus[1:, 1:]
drugVirus = drugVirus.astype(float)

drugNamesSet = {}
for dn in drugNames:
    drugNamesSet[dn] = []

virussim = pd.read_csv(rootDir + files[2], delimiter = ',', header=None).to_numpy()
virussim = virussim[1:, 1:]
virussim = virussim.astype(float)
virussim.shape

drugsim = pd.read_csv(rootDir + files[0], delimiter = ',', header=None).to_numpy()
drugsim = drugsim[1:, 1:]
drugsim = drugsim.astype(float)
drugsim.shape

def customLoss(y_true, y_pred):
    cce = tf.keras.losses.BinaryCrossentropy()
    loss = 0.0
    for i in range(2):
        loss += cce(y_true[:,i], y_pred[:, i]) + (0.25 - (y_pred[:, i] - 0.5) ** 2)

    return loss

N_TRAIN = int(1e4)
STEPS_PER_EPOCH = N_TRAIN//32

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

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
    
    
from sklearn.preprocessing import MinMaxScaler
Y = []
X = []
Z = []

print('SimSize:', drugsim.shape, virussim.shape, drugVirus.shape)
for i in range(len(drugsim)):
    for j in range(1, len(virussim)):
        Y.append(
            drugVirus[i, j]
        )
        Z.append(
            [
                drugNames[i], virusNames[j]
            ]
        )

        X.append(
            np.concatenate(
                (
                    drugsim[i],  virussim[j]
                )
            )
        )

X = np.array(X)
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# X = scaler.fit_transform(X)
Y = np.array(Y)
Z = np.array(Z)
rndIndex = np.random.choice(len(X), len(X), replace = False)
X = X[rndIndex]
Y = Y[rndIndex]
Z = Z[rndIndex]
# X, Y = shuffle(X, Y)
print('DataShape:', X.shape, Y.shape, Z.shape)


from sklearn import metrics
from sklearn.metrics import confusion_matrix

samples = []
for i in range(len(drugsim)):
    samples.append(np.concatenate(
            (
                drugsim[i],  virussim[0]
            )
        )
    )

aucList = []
samples = np.expand_dims(samples, axis = -1)

print('samples:', samples.shape)
for SeT in range(10):
    X, Y = shuffle(X, Y)
    kf = KFold(n_splits=5)
    foldCounter = 1
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        #z_test = Z[test_index]
        
        print('DataShape:', x_train.shape, y_train.shape)
        
        directory = rootDirSave + '/' + str(SeT) + '/' + str(foldCounter) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # np.save(rootDirSave + '/x_train_' + str(foldCounter), x_train)
        # np.save(rootDirSave + '/x_test_' + str(foldCounter), x_test)
        # np.save(rootDirSave + '/y_train_' + str(foldCounter), y_train)
        # np.save(rootDirSave + '/y_test_' + str(foldCounter), y_test)

        checkpoint_filepath = '/tmp/checkpoint/' +str(SeT) +'_'+ str(foldCounter)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_prc',
            mode='max',
            save_best_only=True)


        x_train = np.expand_dims(x_train, axis = -1)
        x_test = np.expand_dims(x_test, axis = -1)

        # pos = len(np.where(y_train == 1)[0])
        # neg = len(np.where(y_train == 0)[0])
        # # initial_bias = np.log([pos/neg])
        # # print(initial_bias)
        # total = pos + neg

        # weight_for_0 = (1 / neg) * (total / 2.0)
        # weight_for_1 = (1 / pos) * (total / 2.0) * 2
        # class_weight = {0: weight_for_0, 1: weight_for_1}

        dModule = buildModule(x_train[0].shape)
        # dModule.summary()

        dModule.fit(x_train, y_train, epochs = 500, batch_size = 32
        ,validation_data=(x_test,y_test), callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=100)])#, class_weight = class_weight)

        dModule.load_weights(checkpoint_filepath)
        
        dModule.save(directory + 'model.h5')
        
        resEval = dModule.evaluate(x_test,y_test)
        y_pred = dModule.predict(x_test)
        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
        auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
        
        np.savetxt(directory + '/y_true.csv', y_test, delimiter=',', fmt='%s')
        np.savetxt(directory + '/y_predicted.csv', y_pred, delimiter=',', fmt='%s')
        

        aucList.append(
            [
                SeT,
                foldCounter,
                auc_keras, 
                resEval[1],
                resEval[2],
                resEval[3],
                resEval[4],
                resEval[5],
                

            ]
        )

        aucList
        np.savetxt(rootDirSave + '/AUC_List.csv', aucList, delimiter=',', fmt='%s')


        res = dModule.predict(samples)
        tempDrugList = []
        for i in range(len(res)):
            if res[i] >= sum(res) / len(res):
                tempDrugList.append(
                    [
                        drugNames[i], res[i][0]
                    ]
                )

        res = res[:,0]
        tempDrugList = np.array(tempDrugList)
        tempDrugList = tempDrugList[tempDrugList[:, 1].argsort()[::-1]]
        
        for dns in range(len(tempDrugList)):
            tempDN = tempDrugList[dns][0]
            tempDNS = [SeT, foldCounter, dns, tempDrugList[dns][1],min(tempDrugList[:, 1].astype(float)),
                       max(tempDrugList[:, 1].astype(float)),
                       sum(tempDrugList[:, 1].astype(float)) / len(tempDrugList[:, 1].astype(float)),
                       len(tempDrugList), min(res), max(res),sum(res) / len(res), len(res)]

            drugNamesSet[tempDN].append(tempDNS)


        for key in drugNamesSet.keys():
            if len(drugNamesSet[key]) > 0:
                np.savetxt(rootDirSaveReper + key + '.csv', drugNamesSet[key], delimiter=',', fmt='%s')

        # np.savetxt(directory + 'CovidResMean_' + str(foldCounter) + '.csv', tempDrugList, delimiter=',', fmt='%s')

        foldCounter += 1