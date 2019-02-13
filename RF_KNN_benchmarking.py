#script to apply Random Forest and K-NN for classification for 4 datasets: Room1.csv, Room2.csv etc
#error is RMSE to evaluate each model
#plot is done for first 60 minutes of records and predicted values in benchmarking


# use glob to select file of the same type (here files with name pattern 'Room*)
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


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
    for i in range(len(data)):
        #print("Dataset is in file: "+ "'" + "{0}'".format(data[i]))
        df = pd.read_csv("./data/"+ data[i])
        
        #RF
        X = df.iloc[:, 0:2].values  
        y = df.iloc[:, 2].values  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) 
        #50 trees in the forest 
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)  
        classifier.fit(X_train, y_train)  
        y_pred = classifier.predict(X_test)
        #RMSE
        errRF = sqrt(mean_squared_error(y_test, y_pred))
        errorRF.append(errRF)
        
        #k-NN
        XX = df.iloc[:, 0:2].values  
        yy = df.iloc[:, 2].values  
        XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size=0.2, random_state=2)  
        #perform normalization 
        #The motivation to use this scaling include robustness to very small standard deviations
        #of features and preserving zero entries in sparse data.
        scaler = StandardScaler()  
        #use 'astype' to convert data with type int64 to float64 for StandardScaler.
        scaler.fit(XX_train.astype(np.float64))
        XX_train = scaler.transform(XX_train.astype(np.float64))  
        XX_test = scaler.transform(XX_test.astype(np.float64))  
        classificator = KNeighborsClassifier(n_neighbors=1000)  
        classificator.fit(XX_train, yy_train)  
        yy_pred = classificator.predict(XX_test)
        #RMSE
        errKNN = sqrt(mean_squared_error(yy_test, yy_pred))
        errorKNN.append(errKNN)
        
        
        #visualization
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize = (12,6))
        t = np.arange(60)
        plt.plot(t,yy_test[:60], label='test values',color='k', linewidth = 2.5)
        plt.plot(t,y_pred[:60], label='predicted values with RF',color='r',linestyle='dashed')
        plt.plot(t,yy_pred[:60], label='predicted values with K-NN',color='darkorange', linestyle='dashed')
        #plt.title('Predicted values versus test data values for Room 1 using RF')
        plt.xlabel('Time steps')
        plt.ylabel('Number of occupants')
        plt.legend()
        plt.savefig('./img/Visualization_RF_vs_K-NN_for_'+'{0}.png'.format(data[i].split(".")[0]), bbox_inches='tight')
        plt.show()
        plt.close('all')
    return errorRF, errorKNN
    
def plot():
    dataset = get_files()
    [e1,e2] = learning(dataset)
    print('RF RMSE for each room is:' ,e1)
    print('K-NN RMSE for each room is :', e2)

    
if __name__ == "__main__":
    plot()
