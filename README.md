# Introduction
The DRaW is a convolutional neural network to predict new virus-antiviral interactions VAIs from approved antivirals. The model is in path “Drug-Repurposing”.  The following figure shows the model architecture. To evaluate VAI, the inputs to the DRaW are a virus similarity vector and antiviral similarity vector. These two vectors are concatenated and the resulting vector is fed to the model. The DRaW consists of four Conv1D layer. The last layer has a sigmoid function as the classifier which returns the probability of virus-antiviral association. 
![Alt text](CovidModel.png?raw=true "CovidModel")
# Running DRaW on COVID-19 datasets
The DRaW has been applied on three COVID-19 datasets, DS1, DS2, and DS3. There are three subdirectories, “DS1_repur”, “DS2_repur”, and “DS3_repur”, in the “Drug-Repurposing” directory. Each subdirectory has been assigned to one of the mentioned datasets. We put the Draw implementation file for each dataset in each subdirectory separately. This is due to keep the corresponding hyperparameters of each dataset. 
We use Adam as the optimizer with a learning rate equal to 0.001, beta1 = 0.9, beta2 = 0.999, and epsilon = 1e−7. The dropout rate is set to 0.5. The batch size is chosen by the number of samples per dataset. This hyperparameter for DS1 is equal to 8, and those for DS2 and DS3 are set to 32.
To run the model, it is enough to execute "Drug-Repurposing.py" script in the command line. After that, execute "score.py". The repurposed drugs will be stored in the "meanScore.csv" spreadsheet. It contains the average of ach drug ranking. The lower, the better. For example, to run the DRaW on DS1:
```bash
cd Drug-Repurposing\DS1_repur
python Drug-Repurposing.py 
python score.py
```
Same goes for other datasets. Just change the directory path.
# Performance analysis
In order to analysis the performance, there is a one extra directory in the root, “Performance_analysis”. By running following command the model is trained on a given dataset and returns its performance metrics, AUC-ROC, AUPR, F1 score, etc.   
The input parameter “dataset_name” is one the following five datasets’ name. The first one iS COVID-19 DS3 and other four are golden benchmarks. 
'DS3','ic','nr','gpcr','e'

```bash
cd Performance_analysis
python main.py dataset_name
```

DRaW: Prediction of COVID-19 Antivirals by Deep Learning -- An Objection on Using Matrix Factorization
M. Hashemi, A. Zabihian, M. Hooshmand, S. Gharaghani

