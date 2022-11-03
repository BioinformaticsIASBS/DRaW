import pandas as pd
import numpy as np



class DataLoader:
    def __init__(self, dataSet):
        
        self.root = 'Dataset/'
        self.dataSet = dataSet
        self.Y = []
        self.X = []
    
    def loader(self):
        
        if self.dataSet == 'covid':
            cs = 1
            intraction = pd.read_csv(self.root + self.dataSet + '/virusdrug.csv', delimiter = ',', header=None, encoding='cp1252').to_numpy()
            intraction = intraction[1:, 1:]
            intraction = intraction.astype(float)

            sim1 = pd.read_csv(self.root + self.dataSet + '/virussim.csv', delimiter = ',', header=None).to_numpy()
            sim1 = sim1[1:, 1:]
            sim1 = sim1.astype(float)
            
            sim2 = pd.read_csv(self.root + self.dataSet + '/drugsim.csv', delimiter = ',', header=None).to_numpy()
            sim2 = sim2[1:, 1:]
            sim2 = sim2.astype(float)
            
        else:
            cs = 0
            intraction = pd.read_csv(self.root + self.dataSet + '/' + self.dataSet +'_admat_dgc.csv', delimiter = ',', header=None, encoding='cp1252').to_numpy()

            intraction = intraction[1:, 1:]
            intraction = intraction.astype(float)

            sim2 = pd.read_csv(self.root + self.dataSet + '/' + self.dataSet +'_simmat_dg.csv', delimiter = ',', header=None).to_numpy()
            sim2 = sim2[1:, 1:]
            sim2 = sim2.astype(float)

            sim1 = pd.read_csv(self.root + self.dataSet + '/' + self.dataSet +'_simmat_dc.csv', delimiter = ',', header=None).to_numpy()
            sim1 = sim1[1:, 1:]
            sim1 = sim1.astype(float)


        for i in range(len(sim2)):
            for j in range(cs, len(sim1)):
                #print(i, j)
                self.Y.append(
                    intraction[i, j]
                )


                self.X.append(
                    np.concatenate(
                        (
                            sim2[i],  sim1[j]
                        )
                    )
                )
            

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        rndIndex = np.random.choice(len(self.X), len(self.X), replace = False)
        self.X = self.X[rndIndex]
        self.Y = self.Y[rndIndex]
        
        return self.X, self.Y

            
            
            
            
            