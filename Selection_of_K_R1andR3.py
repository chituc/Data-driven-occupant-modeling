#script to evaluate k-NN for a given list of values of k
#error is RMSE to evaluate each model


# use glob to select file of the same type (here file with name pattern 'Room')
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os.path




def get_files():
    #get a list of strings with path and file names: ['E:/Github/data\\Room1.csv', ...]
    fileList = glob.glob("./data/Room*")
    dataSetName = []
    for fileName in fileList:
        name = fileName.split("/")[1].split("\\")[1]
        dataSetName.append(name)
    # the returned result is: ['Room1.csv', 'Room2.csv', 'Room3.csv', 'Room4.csv']
    return dataSetName

def learning(data, knn):
    errorKNN = []
    for i in range(len(data)):
        #print("Dataset is in file: "+ "'" + "{0}'".format(data[i]))
        df = pd.read_csv("./data/"+ data[i])
        #k-NN
        XX = df.iloc[:, 0:2].values  
        yy = df.iloc[:, 2].values
        XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size=0.2, random_state=2) 
        #perform normalization 
        #The motivation to use this scaling includes robustness to very small standard deviations
        #of features and preserving zero entries in sparse data.
        scaler = StandardScaler()  
        #use 'astype' to convert data with type int64 to float64 for StandardScaler.
        scaler.fit(XX_train.astype(np.float64))
        XX_train = scaler.transform(XX_train.astype(np.float64))  
        XX_test = scaler.transform(XX_test.astype(np.float64))  
        classificator = KNeighborsClassifier(n_neighbors=knn)  
        classificator.fit(XX_train, yy_train)  
        yy_pred = classificator.predict(XX_test)
        #model evaluation:
        #RMSE
        errKNN = sqrt(mean_squared_error(yy_test, yy_pred))
        #errorKNN.append(round(errKNN,2))
        room = data[i].split(".")[0]
        print('For room {0}, error is {1}, where k is {2} '.format(room,round(errKNN,2),knn))
        with open('RMSE.txt', 'a') as f:
            f.write(str(room) + ',' + str(round(errKNN,2)) + ',' + str(knn) + '\n')
            f.close()
    return errorKNN, room
 

def kSelection():
    #run the K-NN algorithm and writes a file with the result
    kValues = [50,100,200,500,750,1000,1300,1500,1700,2000,2250,2500,2750,3000]
    dataset = get_files()
    for i in kValues:
        learning(dataset, i)
    
    
def getRMSE():
    #it reads from the file with the K-NN results and for each room creates a vector 
    # with the RMSE values, then plots them 
    
    #r1 is the vector containing the RMSE error for Room1
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    
    if(os.path.exists('./RMSE.txt')):
        #add header to dataframe
        df = pd.read_csv('./RMSE.txt', names=['RoomID','RMSE','K'])
        ro1 = df.query("RoomID=='Room1'")['RMSE'].values
        r1.append(ro1)
        r2.append(df.query("RoomID=='Room2'")['RMSE'])
        ro3 = df.query("RoomID=='Room3'")['RMSE'].values
        r3.append(ro3)
        r4.append(df.query("RoomID=='Room4'")['RMSE'])
        #r1.append(df[df['RoomID']=='Room1']['RMSE']) 
        #to get rid or string array, but have only the array from [array([8.05 8.54 ...)]
        return np.array(r1[0]), np.array(r3[0])
 

def plotVisualization(interval):   
    [r1,r3] = getRMSE()
    print('r1: ',r1)
    print('r3: ',r3)
    
    plt.figure(figsize=(12, 6))  
    plt.plot(interval, r1, label='RMSE for Room1', color='purple', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.plot(interval, r3, label='RMSE for Room3', color='darkorange', linestyle ='dashed', marker='o', markerfacecolor='purple', markersize=10)
    plt.xlabel('K value')
    plt.ylabel('RMSE value')
    plt.legend(loc = 'center', fontsize = 'x-large')
    plt.savefig('./img/R1_vs_R3_k_tuning.png', bbox_inches='tight')
    plt.show()
    plt.close('all')
    

   
if __name__ == "__main__":
    #remove the previous file where we are going to write results each time when run the code
    if os.path.exists("./RMSE.txt"):
        os.remove("./RMSE.txt")
    else:
        print("The file 'RMSE.txt' does not exist")
    #to run K-NN for all K values and write the file
    kSelection()
    #to get the RMSE erors in a vector for Room1 and Room3 reading the previous file and plot them
    kValues = [50,100,200,500,750,1000,1300,1500,1700,2000,2250,2500,2750,3000]
    plotVisualization(kValues)

