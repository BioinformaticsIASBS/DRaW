import sys
from dataLoader import DataLoader
from model import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc


if len(sys.argv) != 3:
    print('Missing parameter!')
    exit()
    
dataSetNames = ['covid','ic','nr','gpcr','e']
if sys.argv[1] not in dataSetNames or sys.argv[2] < 1:
    print('Wrong parameter!')
    exit()
    
rootDir = str(sys.argv[1]) + '/'
sets = sys.argv[2]
dataLoader = DataLoader(sys.argv[1])
X, Y = dataLoader.loader()

epochs = 500
if str(sys.argv[1]) == dataSetNames[0] or str(sys.argv[1]) == dataSetNames[3]:
    batch_size = 32
elif str(sys.argv[1]) == dataSetNames[2]:
    batch_size = 8
elif str(sys.argv[1]) == dataSetNames[4]:
    batch_size = 1024
elif str(sys.argv[1]) == dataSetNames[1]:
    batch_size = 256


resList = []
for SeT in range(sets):
    X, Y = shuffle(X, Y)
    kf = KFold(n_splits=5)
    foldCounter = 1
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        directory = rootDir + str(SeT) + '/' + str(foldCounter) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        checkpoint_filepath = '/tmp/checkpoint/' +str(SeT) +'_'+ str(foldCounter)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
        
        x_train = np.expand_dims(x_train, axis = -1)
        x_test = np.expand_dims(x_test, axis = -1)
        
        model = buildModule(x_train[0].shape)
        model.fit(x_train, y_train, epochs = 500, batch_size = batch_size
        ,validation_data=(x_test,y_test), verbose = 0, callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=100)])

        model.load_weights(checkpoint_filepath)
        model.save(directory + 'model.h5')
        
        y_pred = dModule.predict(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        AUC = metrics.auc(fpr, tpr)
        
        precision, recall, threshold = metrics.precision_recall_curve(y_test, y_pred)
        AUPR = metrics.auc(recall,precision)
        
        resList.append(
            [
                SeT, foldCounter, AUC, AUPR
            ]
        )
        print(SeT, foldCounter, AUC, AUPR)
        foldCounter += 1
        
        

    