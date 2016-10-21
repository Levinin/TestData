
import numpy as np
from sklearn.cluster import MeanShift 
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


## Set up some variables out here
myPath = "C:\\Test Data\\"
Data_in = "test data.csv"
testout = "testout.csv"
myPlotPic = "test data.png"
col_int = [0,1,2,3,4]             # List of the columns we are interested in from the file
chart_title = "Test data groups"


## This is getting real data


# Date needs to be in the format of column name on the first row of the column and numeric data in the rest of the fields.
# Pulls in all the data in the file, the col_int above is then used to decide what to actually look at
X = pd.read_csv(myPath + Data_in, header=0, dtype={0: np.float64, 1: np.float64})

print (X. head())



## This is the bit where it fits the data

ms = MeanShift(cluster_all=False)

# Convert the columns of interest to a NumPy array
# Multi-dimensional so could be anything really
msX = np.array(X.iloc[:,col_int])

# print (msX)

ms.fit(msX)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))

# print("Number of estimated clusters:", n_clusters_)
# print(labels)



## Add the labels to the original dataframe and output to csv for analysis

labels_df = pd.DataFrame(labels,columns=['LABELS'])
X = pd.concat([X, labels_df], axis=1)

X.to_csv(myPath + testout)



## This is where we plot the data for viewing

colour= 10 * ["red","blue","yellow","cyan"] + ["purple"]  # Since color and marker -1 will be used as "no group" this makes them always 
markers = 10 * ["^","*","s","o"] + ["."]                  # a purple dot


options = ""
headings = X.columns.tolist()
# Get from the user which items they want to see graphically
# 3D graph so get 3 column indexes
# Can be from any except the labels column which will be the final column number

options = input("Enter 3 columns from the " + str(len(col_int)) + " items as a list \"x,y,z\"(First item is 0) :")

# They should be numbers. Set up some kind of input value test at some point...
options = eval(options)

# This sets up the 3D view in a figure subplot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.set_title(chart_title, fontsize=20,color='blue')

# Set the axis labels to the DataFrame column name
# The column heading of the column of interest of the option chosen for the chart
ax.set_xlabel(headings[col_int[options[0]]],fontsize=14,color='blue')
ax.set_ylabel(headings[col_int[options[1]]],fontsize=14,color='blue')
ax.set_zlabel(headings[col_int[options[2]]],fontsize=14,color='blue')


# Plot all the points in the correct group colour
# The label number will index a colour - unlabelled points will reference -1
# Steps through each point adding it to the scatter graph
for i in range(len(msX)):
    ax.scatter(msX[i][options[0]],
                msX[i][options[1]],
                msX[i][options[2]], 
                c=colour[labels[i]], marker=markers[labels[i]],s=50,zorder=10)

plt.show()



## Output the graph to a png image file for use in a report

plt.savefig(myPath + myPlotPic, bbox_inches='tight')




