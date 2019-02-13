#script to apply Random Forest and K-NN for classification for 4 datasets: first 10 days train and next 2 test
#slide with 1 day: 10 days train and next 2 days for test and tehn continue to slide 1 day
#error is RMSE to evaluate each model
#plot is done for first 60 minutes of records and predicted values in benchmarking


# use glob to select file of the same type (here file with name pattern 'Room')
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from time import time


def get_files():
    #get a list of strings with path and file names: ['D:/Github/data\\Room1.csv', ...]
    fileList = glob.glob("./data/Room*")
    dataSetName = []
    for fileName in fileList:
        name = fileName.split("/")[1].split("\\")[1]
        dataSetName.append(name)
    # the returned result is: ['Room1.csv', 'Room2.csv', 'Room3.csv', 'Room4.csv']
    return dataSetName

def learning(data):
    errorRF = []
    errorKNN = []
    acc_score_RF = []
    acc_score_KNN = []
    RF_training_time = []
    KNN_training_time = []
    for i in range(len(data)):
        #print("Dataset is in file: "+ "'" + "{0}'".format(data[i]))
        df = pd.read_csv("./data/"+ data[i])
        
        #RF
        X = df.iloc[:, 0:2].values  
        y = df.iloc[:, 2].values  
        #days 3-12 are for train (minute wise sampling = > 1440 records/day)
        X_train = X[slice(2880,17280)]
        y_train = y[slice(2880,17280)]
        #days 13-14 are for test
        X_test = X[slice(17280,20160)]
        y_test = y[slice(17280,20160)]
        #50 trees in the forest 
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)  
        t0_RF = time()
        classifier.fit(X_train, y_train)  
        t1_RF = time()
        timp_RF = round(t1_RF-t0_RF, 3)
        RF_training_time.append(timp_RF)
        y_pred = classifier.predict(X_test)
        #model evaluation:
        acc_score_rf = metrics.accuracy_score(y_test, y_pred)
        acc_score_RF.append(acc_score_rf)
        #RMSE
        errRF = sqrt(mean_squared_error(y_test, y_pred))
        errorRF.append(errRF)
        
        
        
        #k-NN
        XX = df.iloc[:, 0:2].values  
        yy = df.iloc[:, 2].values
        #days 3-12 are for train (minute wise sampling = > 1440 records/day)
        XX_train = XX[slice(2880,17280)]
        yy_train = yy[slice(2880,17280)]
        #days 13-14 are for test
        XX_test = XX[slice(17280,20160)]
        yy_test = yy[slice(17280,20160)]
        #perform normalization 
        #The motivation to use this scaling include robustness to very small standard deviations
        #of features and preserving zero entries in sparse data.
        scaler = StandardScaler()  
        #use 'astype' to convert data with type int64 to float64 for StandardScaler.
        scaler.fit(XX_train.astype(np.float64))
        XX_train = scaler.transform(XX_train.astype(np.float64))  
        XX_test = scaler.transform(XX_test.astype(np.float64))  
        classificator = KNeighborsClassifier(n_neighbors=1000)  
        t0_KNN = time()
        classificator.fit(XX_train, yy_train) 
        t1_KNN = time()
        timp_KNN = round(t1_KNN-t0_KNN, 3)
        KNN_training_time.append(timp_KNN)
        yy_pred = classificator.predict(XX_test)
        #model evaluation:
        acc_score_knn = metrics.accuracy_score(yy_test, yy_pred)
        acc_score_KNN.append(acc_score_knn)
        #RMSE
        errKNN = sqrt(mean_squared_error(yy_test, yy_pred))
        errorKNN.append(errKNN)
        
        
        #visualization
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize = (12,6))
        t = np.arange(2880)
        plt.plot(t,y_pred[:2880], label='predicted values with RF',color='r',linestyle='dashed')
        plt.plot(t,y_test[:2880], label='test values',color='k', linewidth = 1.5)
        plt.plot(t,yy_pred[:2880], label='predicted values with K-NN',color='g', linestyle='dashed')
        #plt.title('Predicted values versus test data values for Room 1 using RF')
        plt.xlabel('Time steps')
        plt.ylabel('Number of occupants')
        plt.legend()
        plt.savefig('./img/RF_vs_K-NN_sliding_dataset3_for_'+'{0}.png'.format(data[i].split(".")[0]), bbox_inches='tight')
        plt.show()
        plt.close('all')
    return errorRF, errorKNN, acc_score_RF, acc_score_KNN, RF_training_time, KNN_training_time
    
def plot():
    dataset = get_files()
    [e1, e2, e3, e4, e5, e6] = learning(dataset)
    print('RF RMSE for each room is: ',e1)
    print('K-NN RMSE for each room is :', e2)
    print('RF accuracy score for each room is: ', e3)
    print('K-NN accuracy score for each room is: ', e4)
    print('RF training time for each room is: ', e5)
    print('KNN training time for each room is:', e6)

    
if __name__ == "__main__":
    plot()
