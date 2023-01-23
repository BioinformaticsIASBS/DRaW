import glob2 as glob2
import pandas as pd
import numpy as np
import os


rootDir = ''
rootDirSave = 'models'
rootDirSaveReper = 'drugReper/'
# rootDir = ''
files = [
    'drugsim.csv',
    'virusdrug.csv',
    'virussim.csv',
]

drugVirus = pd.read_csv(rootDir + files[1], delimiter = ',', header=None, encoding='cp1252').to_numpy()

drugNames = drugVirus[1:, 0]

filenames = glob2.glob("drugReper/*.csv")
preDrugs = []
for i in range(len(filenames)):
    preDrugs.append(filenames[i].split('\\')[1].split('.')[0])
meanScore = []
for i in range(len(filenames)):
    sumL = 0
    lenL = 0
    index = pd.read_csv(filenames[i], delimiter = ',', header = None).to_numpy()[:, :3]
    for counter in range(len(index)):
        # if index[counter][0] != 29:
        sumL += index[counter][2]
        lenL += 1
    avg = sumL / lenL
    meanScore.append(
        [
            preDrugs[i],
            avg
        ]
    )

np.savetxt('meanScore_.csv', meanScore, delimiter=',', fmt='%s')