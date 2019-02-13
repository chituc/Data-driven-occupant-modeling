# box_plot.csv contains CO2 values and roomID for all rooms (appended R1,R2,R3,R4)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./data/box_plot.csv")  

fig, ax = plt.subplots()
ax = sns.boxplot(x='roomID', y='CO2[ppm]', data=df, ax=ax, boxprops=dict(alpha=.45))
ax = sns.stripplot(x='roomID', y='CO2[ppm]', data=df, color="darkorange", jitter=0.2, size=2, alpha=.08, ax=ax)
plt.savefig('./img/CO2_boxplot.png')



