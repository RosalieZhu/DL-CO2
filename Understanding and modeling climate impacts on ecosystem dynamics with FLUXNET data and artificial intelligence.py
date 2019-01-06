import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

dataset1 = pd.read_csv('FLX_AT-Neu_DD_2002-2012.csv')
dataset1 = dataset1[['GPP', 'FSDS', 'Prcp', 'T', 'FLDS', 'VPD']]
dataset1 = dataset1.dropna()
dataset1.columns = ['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']
X, y = dataset1[['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']].values, dataset1['GPP'].values

dataset2 = pd.read_csv('FLX_DE-Tha_DD_1996-2014.csv')
dataset2 = dataset2[['GPP', 'FSDS', 'Prcp', 'T', 'FLDS', 'VPD']]
dataset2 = dataset2.dropna()
dataset2.columns = ['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']
X2, y2 = dataset2[['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']].values, dataset2['GPP'].values

dataset3 = pd.read_csv('FLX_NL-Loo_DD_1996-2014.csv')
dataset3 = dataset3[['GPP', 'FSDS', 'Prcp', 'T', 'FLDS', 'VPD']]
dataset3 = dataset3.dropna()
dataset3.columns = ['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']
X3, y3 = dataset3[['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']].values, dataset3['GPP'].values

dataset4 = pd.read_csv('FLX_US-Var_DD_2000-2014.csv')
dataset4 = dataset4[['GPP', 'FSDS', 'Prcp', 'T', 'FLDS', 'VPD']]
dataset4 = dataset4.dropna()
dataset4.columns = ['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']
X4, y4 = dataset4[['GPP', 'Solar Rad', 'Precipitation', 'Temperature', 'Longwave Rad', 'VPD']].values, dataset4['GPP'].values

print('site 1 MAT mean %.1f, std %.1f' %(np.mean(dataset1['Temperature'].values), np.std(dataset1['Temperature'].values)))
print('site 2 MAT mean %.1f, std %.1f' %(np.mean(dataset2['Temperature'].values), np.std(dataset2['Temperature'].values)))
print('site 3 MAT mean %.1f, std %.1f' %(np.mean(dataset3['Temperature'].values), np.std(dataset3['Temperature'].values)))
print('site 4 MAT mean %.1f, std %.1f' %(np.mean(dataset4['Temperature'].values), np.std(dataset4['Temperature'].values)))

print('site 1 MAT mean %.1f, std %.1f' %(np.mean(dataset1['Precipitation'].values)* 24 * 365, np.std(dataset1['Precipitation'].values)* 24 * 365))
print('site 2 MAT mean %.1f, std %.1f' %(np.mean(dataset2['Precipitation'].values)* 24 * 365, np.std(dataset2['Precipitation'].values)* 24 * 365))
print('site 3 MAT mean %.1f, std %.1f' %(np.mean(dataset3['Precipitation'].values)* 24 * 365, np.std(dataset3['Precipitation'].values)* 24 * 365))
print('site 4 MAT mean %.1f, std %.1f' %(np.mean(dataset4['Precipitation'].values)* 24 * 365, np.std(dataset4['Precipitation'].values)* 24 * 365))

print('site 1 MAT mean %.1f, std %.1f' %(np.mean(dataset1['GPP'].values), np.std(dataset1['GPP'].values)))
print('site 2 MAT mean %.1f, std %.1f' %(np.mean(dataset2['GPP'].values), np.std(dataset2['GPP'].values)))
print('site 3 MAT mean %.1f, std %.1f' %(np.mean(dataset3['GPP'].values), np.std(dataset3['GPP'].values)))
print('site 4 MAT mean %.1f, std %.1f' %(np.mean(dataset4['GPP'].values), np.std(dataset4['GPP'].values)))

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig  = plt.figure(figsize=(10,4))
fig.subplots_adjust(top=0.8, wspace = 0.2, hspace = 0.2)
for i in range(dataset1.shape[1]):
    ax = fig.add_subplot(2,3,i+1)
    ax.hist(dataset1.iloc[:,i].values, color = 'white', bins = 15, edgecolor = 'red',linewidth = 1, density = True, alpha = 0.5)
    ax.hist(dataset2.iloc[:,i].values, color = 'white', bins = 15, edgecolor = 'green',linewidth = 1, density = True, alpha = 0.5)
    ax.hist(dataset3.iloc[:,i].values, color = 'white', bins = 15, edgecolor = 'blue',linewidth = 1, density = True, alpha = 0.5)
    ax.hist(dataset4.iloc[:,i].values, color = 'white', bins = 15, edgecolor = 'black',linewidth = 1, density = True, alpha = 0.5)
    
fig  = plt.figure(figsize=(10,4))
fig.subplots_adjust(top=0.8, wspace = 0.2, hspace = 0.2)
for i in range(dataset1.shape[1]):
    ax = fig.add_subplot(2,3,i+1)
    sns.kdeplot(dataset1.iloc[:,i].values, color = 'red',linewidth = 1, alpha = 0.5)
    sns.kdeplot(dataset2.iloc[:,i].values, color = 'green',linewidth = 1, alpha = 0.5)
    sns.kdeplot(dataset3.iloc[:,i].values, color = 'blue',linewidth = 1,  alpha = 0.5)
    sns.kdeplot(dataset4.iloc[:,i].values, color = 'black',linewidth = 1, alpha = 0.5)

sns.pairplot(dataset1)

sns.pairplot(dataset2)

sns.pairplot(dataset3)

sns.pairplot(dataset4)

# scaling
def fit_myscaler(dataset_train,scaler_filename):
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.externals import joblib
    
    scaler = MinMaxScaler(feature_range = (0,1))
    dataset_train_scaled = scaler.fit_transform(dataset_train)
    joblib.dump(scaler,scaler_filename)
    
    return dataset_train_scaled

def get_inverse_myscaler_data(dataset_train, scaler_filename):
    from sklearn.externals import joblib
    
    scaler = joblib.load(scaler_filename)
    dataset_train_rescaled = scaler.inverse_transform(dataset_train)
    
    return dataset_train_rescaled

# build ANN
def deepANN(X,y, nbatchs = 10000, nepochs = 50, nlayers = 3, nnodes = 10, optimize = 'adam', activation_func = 'relu'):
    
    from sklearn.cross_validation import train_test_split
    
    X = fit_myscaler(X,'x_scale')
    y = fit_myscaler(y.reshape(-1,1),'y_scale')
    
    ANN = Sequential()
    ANN.add(Dense(output_dim = nnodes, init = 'uniform', activation = activation_func, input_dim = X.shape[1]))
    for i in range(nlayers -2 ):
        ANN.add(Dense(output_dim = nnodes, init = 'uniform', activation = activation_func))
    ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))
    ANN.compile(optimizer = optimize, loss = 'mse')
    # print(ANN.summary())
    
    X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    
    ANN.fit(X_train,y_train, batch_size = nbatchs, nb_epoch = nepochs)
    
    y_pred = ANN.predict(X_test)
    
    y_test = get_inverse_myscaler_data(y_test.reshape(-1,1),'y_scale')
    y_pred = get_inverse_myscaler_data(y_pred.reshape(-1,1),'y_scale')
    
    return y_test, y_pred

# experiment on batch size
'''  
y_test = []
y_pred = []
for i in [1000,5000,10000]:
    y_test1, y_pred1 = deppANN(np.delete(X,0,axis=1),y, nbatchs = i, nepochs = 500, nlayers = 3, nnodes = 10, optimize = 'adam', activation_func = 'relu')
    
    y_test.append(y_test1)
    y_pred.append(y_pred1)
    
np.savez('FLUXNET_GPP_ANN_batch.npz', y_test=y_test, y_pred=y_pred)

data = np.load('FLUXNET_GPP_ANN_batch.npz')
y_test = data['y_test']
y_pred = data['y_pred']

sns.jointplot(y_test[0], y_pred[0])

sns.jointplot(y_test[1], y_pred[1])

sns.jointplot(y_test[2], y_pred[2])
'''
# Experiment 1 ANN driver
ANN_y_test1_all = []
ANN_y_pred1_all = []
ANN_y_test2_all = []
ANN_y_pred2_all = []
ANN_y_test3_all = []
ANN_y_pred3_all = []
ANN_y_test4_all = []
ANN_y_pred4_all = []
for i in [0,1,2,3,4,5]:
    if i == 0:
        y_test1,y_pred1 = deepANN(X[:,1:],y,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test2,y_pred2 = deepANN(X2[:,1:],y2,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test3,y_pred3 = deepANN(X3[:,1:],y3,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test4,y_pred4 = deepANN(X4[:,1:],y4,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    else:
        y_test1,y_pred1 = deepANN(np.delete(X[:,1:],i-1,axis=1),y,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test2,y_pred2 = deepANN(np.delete(X2[:,1:],i-1,axis=1),y2,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test3,y_pred3 = deepANN(np.delete(X3[:,1:],i-1,axis=1),y3,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test4,y_pred4 = deepANN(np.delete(X4[:,1:],i-1,axis=1),y4,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    ANN_y_test1_all.append(y_test1)
    ANN_y_pred1_all.append(y_pred1)
    ANN_y_test2_all.append(y_test2)
    ANN_y_pred2_all.append(y_pred2)
    ANN_y_test3_all.append(y_test3)
    ANN_y_pred3_all.append(y_pred3)
    ANN_y_test4_all.append(y_test4)
    ANN_y_pred4_all.append(y_pred4)
np.savez('FLUXNET_GPP_ANN.npz',ANN_y_test1_all = ANN_y_test1_all,ANN_y_pred1_all = ANN_y_pred1_all,
         ANN_y_test2_all = ANN_y_test2_all,ANN_y_pred2_all = ANN_y_pred2_all,
         ANN_y_test3_all = ANN_y_test3_all,ANN_y_pred3_all = ANN_y_pred3_all,
         ANN_y_test4_all = ANN_y_test4_all,ANN_y_pred4_all = ANN_y_pred4_all)

# build RNN
def myRNN(X,y,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 50, nlayers = 3, nnodes = 10, optimize = 'adam', activation_func = 'relu'):
    
    X = fit_myscaler(X,'x_scale')
    y = fit_myscaler(y.reshape(-1,1),'y_scale')
    
    # training data
    X_train = []
    y_train = []
    for i in range(mem, int(len(X)*0.8) - pred_mem):
        X_train.append(X[i-mem:i,:])
        y_train.append(y[i:i+pred_mem,0])
    X_train,y_train = np.array(X_train),np.array(y_train)
    
    # buil RNN
    RNN = Sequential()
    RNN.add(LSTM(nnodes, return_sequences = True, activation = activation_func, input_shape = (X_train.shape[1],X_train.shape[2])))
    for i in range(nlayers - 2):
        RNN.add(LSTM(nnodes, return_sequences = True, activation = activation_func))
    RNN.add(LSTM(nnodes, activation = activation_func))
    RNN.add(Dense(units = pred_mem ))
    
    RNN.compile(optimizer = optimize, loss = 'mse')
    RNN.fit(X_train,y_train,epochs =nepochs, batch_size = nbatchs)
    
    # pediction data
    X_test = []
    y_test = []
    for i in range(int(len(X)*0.8)-mem, int(len(X)) - pred_mem):
        X_test.append(X[i-mem:i,:])
        y_test.append(y[i:i+pred_mem,0])
    X_test,y_test = np.array(X_test),np.array(y_test)
    
    y_pred = RNN.predict(X_test)
    
    y_test = get_inverse_myscaler_data(y_test.reshape(-1,1),'y_scale')
    y_pred = get_inverse_myscaler_data(y_pred.reshape(-1,1),'y_scale')
    
    return y_test, y_pred

# RNN experiment, driver
RNN_y_test1_all = []
RNN_y_pred1_all = []
RNN_y_test2_all = []
RNN_y_pred2_all = []
RNN_y_test3_all = []
RNN_y_pred3_all = []
RNN_y_test4_all = []
RNN_y_pred4_all = []
for i in [0,1,2,3,4,5]:
    if i == 0:
        y_test1,y_pred1 = myRNN(X[:,:],y,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test2,y_pred2 = myRNN(X2[:,:],y2,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test3,y_pred3 = myRNN(X3[:,:],y3,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test4,y_pred4 = myRNN(X4[:,:],y4,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    else:
        y_test1,y_pred1 = myRNN(np.delete(X[:,:],i,axis=1),y,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test2,y_pred2 = myRNN(np.delete(X2[:,:],i,axis=1),y2,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test3,y_pred3 = myRNN(np.delete(X3[:,:],i,axis=1),y3,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
        y_test4,y_pred4 = myRNN(np.delete(X4[:,:],i,axis=1),y4,mem=30,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    RNN_y_test1_all.append(y_test1)
    RNN_y_pred1_all.append(y_pred1)
    RNN_y_test2_all.append(y_test2)
    RNN_y_pred2_all.append(y_pred2)
    RNN_y_test3_all.append(y_test3)
    RNN_y_pred3_all.append(y_pred3)
    RNN_y_test4_all.append(y_test4)
    RNN_y_pred4_all.append(y_pred4)
np.savez('FLUXNET_GPP_RNN.npz',RNN_y_test1_all = RNN_y_test1_all,RNN_y_pred1_all = RNN_y_pred1_all,
         RNN_y_test2_all = RNN_y_test2_all,RNN_y_pred2_all = RNN_y_pred2_all,
         RNN_y_test3_all = RNN_y_test3_all,RNN_y_pred3_all = RNN_y_pred3_all,
         RNN_y_test4_all = RNN_y_test4_all,RNN_y_pred4_all = RNN_y_pred4_all)

# RNN experiment, memory
RNN_y_test1_all = []
RNN_y_pred1_all = []
RNN_y_test2_all = []
RNN_y_pred2_all = []
RNN_y_test3_all = []
RNN_y_pred3_all = []
RNN_y_test4_all = []
RNN_y_pred4_all = []
for i in [7,14,21,30,60]:
    y_test1,y_pred1 = myRNN(X[:,:],y,mem=i,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test2,y_pred2 = myRNN(X2[:,:],y2,mem=i,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test3,y_pred3 = myRNN(X3[:,:],y3,mem=i,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test4,y_pred4 = myRNN(X4[:,:],y4,mem=i,pred_mem=1,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    
    RNN_y_test1_all.append(y_test1)
    RNN_y_pred1_all.append(y_pred1)
    RNN_y_test2_all.append(y_test2)
    RNN_y_pred2_all.append(y_pred2)
    RNN_y_test3_all.append(y_test3)
    RNN_y_pred3_all.append(y_pred3)
    RNN_y_test4_all.append(y_test4)
    RNN_y_pred4_all.append(y_pred4)

np.savez('FLUXNET_GPP_RNN_mem.npz',RNN_y_test1_all = RNN_y_test1_all,RNN_y_pred1_all = RNN_y_pred1_all,
         RNN_y_test2_all = RNN_y_test2_all,RNN_y_pred2_all = RNN_y_pred2_all,
         RNN_y_test3_all = RNN_y_test3_all,RNN_y_pred3_all = RNN_y_pred3_all,
         RNN_y_test4_all = RNN_y_test4_all,RNN_y_pred4_all = RNN_y_pred4_all)


# RNN experiment, prediction memory
RNN_y_test1_all = []
RNN_y_pred1_all = []
RNN_y_test2_all = []
RNN_y_pred2_all = []
RNN_y_test3_all = []
RNN_y_pred3_all = []
RNN_y_test4_all = []
RNN_y_pred4_all = []
for i in [1,3,5,7,14]:
    y_test1,y_pred1 = myRNN(X[:,:],y,mem=30,pred_mem=i,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test2,y_pred2 = myRNN(X2[:,:],y2,mem=30,pred_mem=i,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test3,y_pred3 = myRNN(X3[:,:],y3,mem=30,pred_mem=i,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    y_test4,y_pred4 = myRNN(X4[:,:],y4,mem=30,pred_mem=i,nbatchs = 10000, nepochs = 100, nlayers =3, nnodes = 10, optimize = 'adam',activation_func = 'relu')
    
    RNN_y_test1_all.append(y_test1)
    RNN_y_pred1_all.append(y_pred1)
    RNN_y_test2_all.append(y_test2)
    RNN_y_pred2_all.append(y_pred2)
    RNN_y_test3_all.append(y_test3)
    RNN_y_pred3_all.append(y_pred3)
    RNN_y_test4_all.append(y_test4)
    RNN_y_pred4_all.append(y_pred4)

np.savez('FLUXNET_GPP_RNN_predmem.npz',RNN_y_test1_all = RNN_y_test1_all,RNN_y_pred1_all = RNN_y_pred1_all,
         RNN_y_test2_all = RNN_y_test2_all,RNN_y_pred2_all = RNN_y_pred2_all,
         RNN_y_test3_all = RNN_y_test3_all,RNN_y_pred3_all = RNN_y_pred3_all,
         RNN_y_test4_all = RNN_y_test4_all,RNN_y_pred4_all = RNN_y_pred4_all)

# visialization
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
data = np.load('FLUXNET_GPP_ANN.npz')
ANN_y_test1_all = data['ANN_y_test1_all']
ANN_y_pred1_all = data['ANN_y_pred1_all']
ANN_y_test2_all = data['ANN_y_test2_all']
ANN_y_pred2_all = data['ANN_y_pred2_all']
ANN_y_test3_all = data['ANN_y_test3_all']
ANN_y_pred3_all = data['ANN_y_pred3_all']
ANN_y_test4_all = data['ANN_y_test4_all']
ANN_y_pred4_all = data['ANN_y_pred4_all']

mse_1 = np.full([6],np.nan)
mse_2 = np.full([6],np.nan)
mse_3 = np.full([6],np.nan)
mse_4 = np.full([6],np.nan)
for i in range(6):
    mse_1[i] = np.nanmean((ANN_y_test1_all[i] - ANN_y_pred1_all[i])**2)
    mse_2[i] = np.nanmean((ANN_y_test2_all[i] - ANN_y_pred2_all[i])**2)
    mse_3[i] = np.nanmean((ANN_y_test3_all[i] - ANN_y_pred3_all[i])**2)
    mse_4[i] = np.nanmean((ANN_y_test4_all[i] - ANN_y_pred4_all[i])**2)

correlation_1 = np.full([6],np.nan)
correlation_2 = np.full([6],np.nan)
correlation_3 = np.full([6],np.nan)
correlation_4 = np.full([6],np.nan)
for i in range(6):
    [correlation_1[i], pvalue] = pearsonr(ANN_y_test1_all[i],ANN_y_pred1_all[i])
    [correlation_2[i], pvalue] = pearsonr(ANN_y_test2_all[i],ANN_y_pred2_all[i])
    [correlation_3[i], pvalue] = pearsonr(ANN_y_test3_all[i],ANN_y_pred3_all[i])
    [correlation_4[i], pvalue] = pearsonr(ANN_y_test4_all[i],ANN_y_pred4_all[i])

fig,ax = plt.subplots(1,2,figsize=(12,6))
ind = np.array([1,2,3,4,5,6])
p1 = ax[0].bar(ind, mse_1,width = 0.2,color='green')
p2 = ax[0].bar(ind+0.2, mse_2,width = 0.2,color='blue')
p3 = ax[0].bar(ind+0.4, mse_3,width = 0.2,color='orange')
p4 = ax[0].bar(ind+0.6, mse_4,width = 0.2,color='red')
ax[0].set_title('Predicted mean square error')
ax[0].set_xticks(ind+0.3)
ax[0].set_xticklabels(('all','no FSDS', 'no Pre', 'no Temp', 'no FLDS','no VPD'))
ax[0].legend((p1[0],p2[0],p3[0],p4[0]),('FLX_AT-Neu','FLX_DE-Tha','FLX_NL-Loo','FLX_US-Var'))

p1 = ax[1].bar(ind, correlation_1,width = 0.2,color='green')
p2 = ax[1].bar(ind+0.2, correlation_2,width = 0.2,color='blue')
p3 = ax[1].bar(ind+0.4, correlation_3,width = 0.2,color='orange')
p4 = ax[1].bar(ind+0.6, correlation_4,width = 0.2,color='red')
ax[1].set_title('Correlation between prediction and observations')
ax[1].set_xticks(ind+0.3)
ax[1].set_xticklabels(('all','no FSDS', 'no Pre', 'no Temp', 'no FLDS','no VPD'))
plt.savefig('FLUXNET_GPP.ANN_prediction.png',dpi=600)

# RNN
data = np.load('FLUXNET_GPP_RNN.npz')
RNN_y_test1_all = data['RNN_y_test1_all']
RNN_y_pred1_all = data['RNN_y_pred1_all']
RNN_y_test2_all = data['RNN_y_test2_all']
RNN_y_pred2_all = data['RNN_y_pred2_all']
RNN_y_test3_all = data['RNN_y_test3_all']
RNN_y_pred3_all = data['RNN_y_pred3_all']
RNN_y_test4_all = data['RNN_y_test4_all']
RNN_y_pred4_all = data['RNN_y_pred4_all']

mse_1 = np.full([6],np.nan)
mse_2 = np.full([6],np.nan)
mse_3 = np.full([6],np.nan)
mse_4 = np.full([6],np.nan)
for i in range(6):
    mse_1[i] = np.nanmean((RNN_y_test1_all[i] - RNN_y_pred1_all[i])**2)
    mse_2[i] = np.nanmean((RNN_y_test2_all[i] - RNN_y_pred2_all[i])**2)
    mse_3[i] = np.nanmean((RNN_y_test3_all[i] - RNN_y_pred3_all[i])**2)
    mse_4[i] = np.nanmean((RNN_y_test4_all[i] - RNN_y_pred4_all[i])**2)

correlation_1 = np.full([6],np.nan)
correlation_2 = np.full([6],np.nan)
correlation_3 = np.full([6],np.nan)
correlation_4 = np.full([6],np.nan)
for i in range(6):
    [correlation_1[i], pvalue] = pearsonr(RNN_y_test1_all[i],RNN_y_pred1_all[i])
    [correlation_2[i], pvalue] = pearsonr(RNN_y_test2_all[i],RNN_y_pred2_all[i])
    [correlation_3[i], pvalue] = pearsonr(RNN_y_test3_all[i],RNN_y_pred3_all[i])
    [correlation_4[i], pvalue] = pearsonr(RNN_y_test4_all[i],RNN_y_pred4_all[i])

fig,ax = plt.subplots(1,2,figsize=(12,6))
ind = np.array([1,2,3,4,5,6])
p1 = ax[0].bar(ind, mse_1,width = 0.2,color='green')
p2 = ax[0].bar(ind+0.2, mse_2,width = 0.2,color='blue')
p3 = ax[0].bar(ind+0.4, mse_3,width = 0.2,color='orange')
p4 = ax[0].bar(ind+0.6, mse_4,width = 0.2,color='red')
ax[0].set_title('Predicted mean square error')
ax[0].set_xticks(ind+0.3)
ax[0].set_xticklabels(('all','no FSDS', 'no Pre', 'no Temp', 'no FLDS','no VPD'))
ax[0].legend((p1[0],p2[0],p3[0],p4[0]),('FLX_AT-Neu','FLX_DE-Tha','FLX_NL-Loo','FLX_US-Var'))

p1 = ax[1].bar(ind, correlation_1,width = 0.2,color='green')
p2 = ax[1].bar(ind+0.2, correlation_2,width = 0.2,color='blue')
p3 = ax[1].bar(ind+0.4, correlation_3,width = 0.2,color='orange')
p4 = ax[1].bar(ind+0.6, correlation_4,width = 0.2,color='red')
ax[1].set_title('Correlation between prediction and observations')
ax[1].set_xticks(ind+0.3)
ax[1].set_xticklabels(('all','no FSDS', 'no Pre', 'no Temp', 'no FLDS','no VPD'))

plt.savefig('FLUXNET_GPP.RNN_prediction.png',dpi=600)

#
data = np.load('FLUXNET_GPP_RNN_mem.npz')
RNN_y_test1_all = data['RNN_y_test1_all']
RNN_y_pred1_all = data['RNN_y_pred1_all']
RNN_y_test2_all = data['RNN_y_test2_all']
RNN_y_pred2_all = data['RNN_y_pred2_all']
RNN_y_test3_all = data['RNN_y_test3_all']
RNN_y_pred3_all = data['RNN_y_pred3_all']
RNN_y_test4_all = data['RNN_y_test4_all']
RNN_y_pred4_all = data['RNN_y_pred4_all']

mse_1 = np.full([5],np.nan)
mse_2 = np.full([5],np.nan)
mse_3 = np.full([5],np.nan)
mse_4 = np.full([5],np.nan)
for i in range(5):
    mse_1[i] = np.nanmean((RNN_y_test1_all[i] - RNN_y_pred1_all[i])**2)
    mse_2[i] = np.nanmean((RNN_y_test2_all[i] - RNN_y_pred2_all[i])**2)
    mse_3[i] = np.nanmean((RNN_y_test3_all[i] - RNN_y_pred3_all[i])**2)
    mse_4[i] = np.nanmean((RNN_y_test4_all[i] - RNN_y_pred4_all[i])**2)

correlation_1 = np.full([5],np.nan)
correlation_2 = np.full([5],np.nan)
correlation_3 = np.full([5],np.nan)
correlation_4 = np.full([5],np.nan)
for i in range(5):
    [correlation_1[i], pvalue] = pearsonr(RNN_y_test1_all[i],RNN_y_pred1_all[i])
    [correlation_2[i], pvalue] = pearsonr(RNN_y_test2_all[i],RNN_y_pred2_all[i])
    [correlation_3[i], pvalue] = pearsonr(RNN_y_test3_all[i],RNN_y_pred3_all[i])
    [correlation_4[i], pvalue] = pearsonr(RNN_y_test4_all[i],RNN_y_pred4_all[i])

fig,ax = plt.subplots(1,2,figsize=(12,6))
ind = np.array([1,2,3,4,5])
p1 = ax[0].bar(ind, mse_1,width = 0.2,color='green')
p2 = ax[0].bar(ind+0.2, mse_2,width = 0.2,color='blue')
p3 = ax[0].bar(ind+0.4, mse_3,width = 0.2,color='orange')
p4 = ax[0].bar(ind+0.6, mse_4,width = 0.2,color='red')
ax[0].set_title('Predicted mean square error')
ax[0].set_xticks(ind+0.3)
ax[0].set_xticklabels(('7 days','14 days', '21 days', '30 days', '60 days'))
ax[0].legend((p1[0],p2[0],p3[0],p4[0]),('FLX_AT-Neu','FLX_DE-Tha','FLX_NL-Loo','FLX_US-Var'))

p1 = ax[1].bar(ind, correlation_1,width = 0.2,color='green')
p2 = ax[1].bar(ind+0.2, correlation_2,width = 0.2,color='blue')
p3 = ax[1].bar(ind+0.4, correlation_3,width = 0.2,color='orange')
p4 = ax[1].bar(ind+0.6, correlation_4,width = 0.2,color='red')
ax[1].set_title('Correlation between prediction and observations')
ax[1].set_xticks(ind+0.3)
ax[1].set_xticklabels(('7 days','14 days', '21 days', '30 days', '60 days'))

plt.savefig('FLUXNET_GPP.RNN_mem_prediction.png',dpi=600)

#
data = np.load('FLUXNET_GPP_RNN_predmem.npz')
RNN_y_test1_all = data['RNN_y_test1_all']
RNN_y_pred1_all = data['RNN_y_pred1_all']
RNN_y_test2_all = data['RNN_y_test2_all']
RNN_y_pred2_all = data['RNN_y_pred2_all']
RNN_y_test3_all = data['RNN_y_test3_all']
RNN_y_pred3_all = data['RNN_y_pred3_all']
RNN_y_test4_all = data['RNN_y_test4_all']
RNN_y_pred4_all = data['RNN_y_pred4_all']

mse_1 = np.full([5],np.nan)
mse_2 = np.full([5],np.nan)
mse_3 = np.full([5],np.nan)
mse_4 = np.full([5],np.nan)
for i in range(5):
    mse_1[i] = np.nanmean((RNN_y_test1_all[i] - RNN_y_pred1_all[i])**2)
    mse_2[i] = np.nanmean((RNN_y_test2_all[i] - RNN_y_pred2_all[i])**2)
    mse_3[i] = np.nanmean((RNN_y_test3_all[i] - RNN_y_pred3_all[i])**2)
    mse_4[i] = np.nanmean((RNN_y_test4_all[i] - RNN_y_pred4_all[i])**2)

correlation_1 = np.full([5],np.nan)
correlation_2 = np.full([5],np.nan)
correlation_3 = np.full([5],np.nan)
correlation_4 = np.full([5],np.nan)
for i in range(5):
    [correlation_1[i], pvalue] = pearsonr(RNN_y_test1_all[i],RNN_y_pred1_all[i])
    [correlation_2[i], pvalue] = pearsonr(RNN_y_test2_all[i],RNN_y_pred2_all[i])
    [correlation_3[i], pvalue] = pearsonr(RNN_y_test3_all[i],RNN_y_pred3_all[i])
    [correlation_4[i], pvalue] = pearsonr(RNN_y_test4_all[i],RNN_y_pred4_all[i])

fig,ax = plt.subplots(1,2,figsize=(12,6))
ind = np.array([1,2,3,4,5])
p1 = ax[0].bar(ind, mse_1,width = 0.2,color='green')
p2 = ax[0].bar(ind+0.2, mse_2,width = 0.2,color='blue')
p3 = ax[0].bar(ind+0.4, mse_3,width = 0.2,color='orange')
p4 = ax[0].bar(ind+0.6, mse_4,width = 0.2,color='red')
ax[0].set_title('Predicted mean square error')
ax[0].set_xticks(ind+0.3)
ax[0].set_xticklabels(('1 days','7 days','14 days', '21 days', '30 days'))
ax[0].legend((p1[0],p2[0],p3[0],p4[0]),('FLX_AT-Neu','FLX_DE-Tha','FLX_NL-Loo','FLX_US-Var'))

ind = np.array([1,2,3,4,5])
p1 = ax[1].bar(ind, correlation_1,width = 0.2,color='green')
p2 = ax[1].bar(ind+0.2, correlation_2,width = 0.2,color='blue')
p3 = ax[1].bar(ind+0.4, correlation_3,width = 0.2,color='orange')
p4 = ax[1].bar(ind+0.6, correlation_4,width = 0.2,color='red')
ax[1].set_title('Correlation between prediction and observations')
ax[1].set_xticks(ind+0.3)
ax[1].set_xticklabels(('1 days','7 days','14 days', '21 days', '30 days'))

plt.savefig('FLUXNET_GPP.RNN_predmem_prediction.png',dpi=600)
#plt.tight_layout()

