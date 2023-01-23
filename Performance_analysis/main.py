import sys
from dataLoader import DataLoader
from model import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics


if len(sys.argv) != 2:
    print('Missing parameter!')
    exit()
    
dataSetNames = ['DS3','ic','nr','gpcr','e']
if sys.argv[1] not in dataSetNames:
    print('Wrong parameter!')
    exit()

folds = 5
rootDir = str(sys.argv[1]) + '/'
sets = 1
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

        np.savetxt(directory + '/y_true.csv', y_test, delimiter=',', fmt='%s')
        np.savetxt(directory + '/y_predicted.csv', y_pred, delimiter=',', fmt='%s')

        



for s in range(sets):
    for fold in range(1, folds + 1):
        directory = rootDir + str(s) + '/' + str(fold) 
        trueLable = pd.read_csv(directory + '/y_true.csv', ).to_numpy()
        predictedLable = pd.read_csv(directory + '/y_predicted.csv', ).to_numpy()
        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = metrics.roc_curve(trueLable, predictedLable)
        auc = metrics.auc(nn_fpr_keras, nn_tpr_keras)   #AUC

        precision, recall, thresholds = metrics.precision_recall_curve(trueLable, predictedLable)
        aupr = metrics.auc(recall, precision)   #AUPR

        gmeans = np.sqrt(nn_tpr_keras * (1-nn_fpr_keras))
        ix = np.argmax(gmeans)
        gmeansMax = gmeans[ix] #Gmean
        gmeansTR = nn_thresholds_keras[ix] #Gmean TR
        
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        f1TR = thresholds[ix] 
        f1Max = fscore[ix] #F1Max

        resTemp = np.zeros(
                (
                    len(predictedLable),
                )
            )
        resTemp[np.where(predictedLable[:,0] >= gmeansTR)] = 1

        if math.isnan(f1Max) == False:
            maxPRE = precision[ix]  #precision
            maxREc = recall[ix]     #recall


        else:
            
            f1Max = metrics.f1_score(trueLable, resTemp)
            maxPRE = metrics.precision_score(trueLable, resTemp)
            maxREc = metrics.recall_score(trueLable, resTemp)

        tn, fp, fn, tp = metrics.confusion_matrix(trueLable, resTemp).ravel()
        specificity = tn / (tn+fp)  #specificity
        
        # nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = metrics.roc_curve(recall, precision)
        aupr = metrics.auc(recall, precision)  #AUPR

        temp =[
                fold,
                maxREc,
                specificity,
                maxPRE,
                f1Max,
                auc,
                aupr,
            ]
        resList.append(
            temp
        )
        # print(resList.shape)

        np.savetxt('Mes.csv', np.array(resList), delimiter=',')