# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import csv
import statistics
path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/Lab 5 ' \
       'Neural Networks/'

# Data import
glob = dict()       
with open(path +'covid19_global_dataset.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile)
    for r in datareader:
        c = r[0]
        if c=='week':
            weekID = r[1:]
        else:
            tmp = []
            for i in range(0,21):
                tmp.append(0)
            darray = r[1:]
            for i in range(0,len(darray)):
                t = int(weekID[i])
                d = int(darray[i])
                if t<21:
                    tmp[t] += d
            glob[c] = tmp    

# New Cases
allNews = []
for c in glob:
    tmp = glob[c]
    tmp2 = [tmp[0]]
    allNews.append(tmp[0])
    for i in range(1,len(tmp)):
        tmp2.append(tmp[i] - tmp[i-1])
        allNews.append(tmp[i] - tmp[i-1])
    glob[c] = tmp2

# Build Dataset
# focus on only US, Spain, Italy and Canada for test
country_list = ['US', 'Spain', 'Italy', 'Canada']
X_test = []
Y_test = []
step = 10
for c in glob:
    if c in country_list:
        print(c)
        tmp = glob[c]
        for j in range(0,len(tmp)-step-1):
            stest = sum(tmp[j:j+step])
            if stest>0:
                X_test.append(tmp[j:j+step])
                Y_test.append(tmp[j+step])

npX_test = np.array(X_test)
npY_test = np.array(Y_test)

X= []
Y= []
step = 10
for c in glob:
    if c not in country_list:
        tmp = glob[c]
        for j in range(0,len(tmp)-step-1):
            stest = sum(tmp[j:j+step])
            if stest>0:
                X.append(tmp[j:j+step])
                Y.append(tmp[j+step])


npX = np.array(X)
npY = np.array(Y)

from sklearn.model_selection import train_test_split
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(npX, npY, test_size=0.33)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

#cv5 = KFold(n_splits=5, random_state=1, shuffle=True)
#################
#Comparing Models
#################

#S1: Two hidden layers with 15 neurons in each layer (current setting)

clf = MLPRegressor(hidden_layer_sizes=(15,15), random_state=1,max_iter=2000)
scores = cross_val_score(clf, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared error =', scores)
#R-Squared error = 0.7834369450220723

#S2: Two hidden layers with 10 neurons in each layer
clf2 = MLPRegressor(hidden_layer_sizes=(10,10), random_state=1,max_iter=2000)
scores2 = cross_val_score(clf2, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared error =', scores2)
# R-Squared = 0.8062416689890292

#S3: Two hidden layers with 10 neurons in the first layer and 15 neurons in the second layer
clf3 = MLPRegressor(hidden_layer_sizes=(10,15), random_state=1,max_iter=2000)
scores3 = cross_val_score(clf3, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores3)
# R-Squared = 0.7980480078053006

#S4: Two hidden layers with 15 neurons in the first layer and 10 neurons in the second layer
clf4 = MLPRegressor(hidden_layer_sizes=(15,10), random_state=1,max_iter=2000)
scores4 = cross_val_score(clf4, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores4)
# R-Squared = 0.7936494907864421

#S5: Three hidden layers with 15 neurons in each layer
clf5 = MLPRegressor(hidden_layer_sizes=(15,15,15), random_state=1,max_iter=2000)
scores5= cross_val_score(clf5, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores5)
# R-Squared = 0.754636173423503


#S6: Three hidden layers with 10 neurons in each layer
clf6 = MLPRegressor(hidden_layer_sizes=(10,10,10), random_state=1,max_iter=2000)
scores6= cross_val_score(clf6, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores6)
# R-Squared = 0.7805797336713728

#S7: One hidden layer with 10 neurons
clf7 = MLPRegressor(hidden_layer_sizes=(10), random_state=1,max_iter=2000)
scores7= cross_val_score(clf7, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores7)
# R-Squared = 0.8004780772684914

#S8: One hidden layer with 15 neurons
clf8 = MLPRegressor(hidden_layer_sizes=(15), random_state=1,max_iter=2000)
scores8= cross_val_score(clf8, npX, npY, scoring='r2', cv=5, n_jobs=-1).mean()
print('R-Squared =', scores8)
# R-Squared = 0.8101451829246592


#############
#now use the best model (S8) on the test set:
clf8.fit(npX, npY)
pred_y = clf8.predict(npX_test)
# get the mean squared error:
mse_8 = mean_squared_error(npY_test, pred_y)
print(mse_8)
#16900688190.776178

mae_8 = mean_absolute_error(npY_test, pred_y)
print(mae_8)
# 70705.47379932256

mae_list = []
#Mean Absolute Error For Each Country:
for i in range (0, len(pred_y)):
    dif = abs(pred_y[i]-npY_test[i])
    mae_list.append(dif)

country_dict = {'Canada':[], 'Italy':[], 'Spain':[], 'US':[]}

lister=[]
for j in range(0,4):
    mini = []
    lister.append(mini)
    for i in range(j, 40, 4):
        mini.append(mae_list[i])

print(lister)

country_keys = list(country_dict.keys())

for index, key in enumerate(country_dict.keys()):
    country_dict[key] = statistics.mean(lister[index])


#Bar Chart:
country = list(country_dict.keys())
values = list(country_dict.values())

plt.bar(range(len(country_dict)), values, tick_label=country)
plt.title('Mean Aboslute Error Per Country')
plt.show()

# Put this on the already created Git Hub.