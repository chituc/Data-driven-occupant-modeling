#random forest for classification
#this script save an image with a tree for data from Room3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pydot
import sklearn 


dataset = pd.read_csv("./data/Room3.csv")  

X = dataset.iloc[:, 0:2].values  
y = dataset.iloc[:, 2].values  

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)  
#training the algorithm
classifier = RandomForestClassifier(n_estimators=3, random_state=42, max_depth=2)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test) 

#find features importance
importances = list(classifier.feature_importances_)
print('importance',importances)
feature_list=['CO2','Damper']
#manual labeled; from the data owners I knew the max no of occupants is 67
class_list = ['1','2','0','3','4','5','6','7','8','9','10','11','12','13',
              '14','15','16','17','18','19','20','21','22','23','24','25',
              '26','27','28','29','30','31','32','33','34','35','36','37',
              '38','39','40','41','42','43','44','45','46','47','48','49',
              '50','51','52','53','54','55','56','57','58','59','60','61',
              '62','63','64','65','66','67']
y_pos = range(len(feature_list))
#beside the root node, there are only two other layers
tree = classifier.estimators_[2]
sklearn.tree.export_graphviz(tree, out_file = 'tree.dot', 
                feature_names = feature_list,
                class_names=class_list, 
                rounded = True, precision = 2, filled=True)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('./img/DecisionTree_Room3_graph.png')