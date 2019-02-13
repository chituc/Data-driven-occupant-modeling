#plot RF performance vs KNN
#these vectors with the results are collected by running the scripts
#sliding_1_RF_KNN.py, sliding_2_RF_KNN.py etc


import matplotlib.pyplot as plt
import numpy as np


fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize = (12,6))
t = np.arange(2879)
#Room 1 for all sliding datasets:
RF1 = [4.75, 4.82, 17.17, 18.13]
KNN1 = [4.52, 4.52, 11.04, 12.56]

#Room 2 for all sliding datasets:
RF2 = [2.76, 2.74, 3.47, 4.62]
KNN2 = [3.28, 3.16, 3.54, 4.72]

#Room 3 for all sliding datasets
RF3 = [2.50, 1.7, 2.16, 3.02  ]
KNN3 = [1.70, 1.38, 3.03, 4.72 ]

#Room 4 for all sliding datasets:
RF4 = [4.18, 4.18, 1.08, 2.98 ]
KNN4 = [4.64, 4.64, 1.27, 4.25 ]
#t = np.arange(4)
t = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4']
plt.plot(t,RF1, label='RF evaluation for Room1',color='mediumseagreen',linestyle='dashed',marker = 's', markerfacecolor = 'k')
plt.plot(t,KNN1, label='K-NN evaluation for Room1',color='mediumseagreen', marker = 's', markerfacecolor = 'k')
plt.plot(t,RF2, label='RF evaluation for Room2',color='sienna',linestyle='dashed',marker = 's', markerfacecolor = 'k')
plt.plot(t,KNN2, label='K-NN evaluation for Room2',color='sienna', marker = 's', markerfacecolor = 'k')
plt.plot(t,RF3, label='RF evaluation for Room3',color='darkorange',linestyle='dashed',marker = 's', markerfacecolor = 'k')
plt.plot(t,KNN3, label='K-NN evaluation for Room3',color='darkorange', marker = 's', markerfacecolor = 'k')
plt.plot(t,RF4, label='RF evaluation for Room4',color='royalblue',linestyle='dashed',marker = 's', markerfacecolor = 'k')
plt.plot(t,KNN4, label='K-NN evaluation for Room4',color='royalblue', marker = 's', markerfacecolor = 'k')
plt.xlabel('Datasets using sliding window split')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('./img/RF_Versus_KNN_RMSE_with_sliding_window_data.png', bbox_inches='tight')
plt.show()
plt.close('all')

