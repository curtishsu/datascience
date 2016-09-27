#KNN & Linear Regression Modeling and Imputation

import numpy as np
import pandas as pd
import random
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.linear_model import LinearRegression as Lin_Reg
import time
import matplotlib
import matplotlib.pyplot as plt
import time
#%matplotlib inline
import matplotlib.mlab as mlab
import math
import scipy

#Problem 1: Implementing KNN and Linear Regression Modeling from scratch and by using the SKlearn 
#Split the data in training and testing groups, fits the training group, predicts testing, and scores accuracy
#Execute with a data set and time the efficiency of each function

#Declare a function that will create one group by randomly selecting a fraction of a larger group=
#and then dropping those values to form a new group
def split(data, m):
    dec = m * .01
    train = data.sample(frac = dec, replace=False)
    test = data.drop(train.index)
    return train, test

#Declare function KNN predict
#Input k, which is number of nearest neighbors, and training and testing groups
def knn_predict(k, training, testing):
    
    knn_start = time.time()
    #Create column to add predictions in
    testing = pd.DataFrame(testing.ix[:,0])
    predictions = []
    
    #Loop through all of testing to find 'y' values of nearest neighbors
    for index, test_row in testing.iterrows():
        distances = []
        y_values = []
        
        #Loop through all of train and store in distances the distance and it's associated 'y' values
        for index, train_row in training.iterrows():
            dist = abs(train_row[0] - test_row[0])
            distances.append(dist)
            y_values.append(train_row[1])
        
        #Sort the distances and save the k values
        distance_y_list = pd.DataFrame({'distance':distances, 'y':y_values})
        distance_y_list.sort_values('distance', inplace=True)
        neighbors = distance_y_list[0:k]

        prediction_means = np.mean(neighbors.ix[:,1])
        predictions.append(prediction_means)
            
    #save predictions into a new column of test
    testing['knn_predictions'] = predictions
    
    knn_time = time.time() - knn_start
    return testing, knn_time



#Define a function that inputs a training group and returns a slope and coefficient (and run time)
def linear_reg_fit(train):
    
    lin_fit_start = time.time()
    
    #Create an array for the x variable and the y variable
    train_x = train.iloc[:,0].as_matrix()
    train_y = train.iloc[:,1].as_matrix()
    
    #Calculate slope and the constant
    slope = np.sum((train_x - np.mean(train_x)) * (train_y - np.mean(train_y)))/np.sum((train_x - np.mean(train_x))**2)
    constant = np.mean(train_y) - slope*np.mean(train_x)
    
    lin_fit_time = time.time() - lin_fit_start
    return slope, constant, lin_fit_time

#Define a function that inputs slope, an intercept, and an nx1 df, and return the dataframe with predicted values
def lin_reg_predict(test, slope, constant):
    
    lin_predict_start = time.time()
    #Create predictions to store all predicated values
    predictions= []
    test = pd.DataFrame(test.ix[:,0])
    test['lin_reg_predicted'] = ""
    
    #Loop through each row and fit the x values; store in predictions
    for index, test_row in test.iterrows():
        predicted_y = slope * test_row[0] + constant
        predictions.append(predicted_y)
    test['lin_reg_predicted'] = predictions
    
    lin_predict_time = time.time() - lin_predict_start
    return test, lin_predict_time



#Define a function that inputs predicted values and actual values, and return the R^2 value
def score(predicted, actual):
    
    score_start_time = time.time()
    #Create tss(total sum of squares) and rss(residual sum of squares) to find R^2
    tss = np.sum((actual.iloc[:,1] - np.mean(actual.iloc[:,1]))**2)
    rss = np.sum((actual.iloc[:,1] - predicted.iloc[:,1])**2)
    R2 = 1 - (rss/tss)
    score_time = time.time() - score_start_time
    
    return R2, score_time



#Execute using the functions we created
#Import dataset one and split it into 70% training 30% testing; splice testing so it only has one column
split_start_time = time.time()
dataset_1_full = pd.read_csv('dataset_1_full.txt')
train, actual_test = split(dataset_1_full, 70)
testing = pd.DataFrame(actual_test.ix[:,0])
split_time = time.time() - split_start_time

#Run KNN predictions and print out 
knn_run = time.time()
knn_predictions = knn_predict(30, train, testing)
accuracy_run = score(knn_predictions[0], actual_test)
total_knn_run = time.time() - knn_run + split_time
print "The R^2 value of our KNN model is " + str(accuracy_run[0])
print "The run time of KNN regressing is "  +str(total_knn_run) + " seconds"


#Run Linear Regression fitting to get the slope and constant, and then run the model; print out accuracy
lin_run = time.time()
slope, constant, fit_time = linear_reg_fit(train)
lin_predict, predict_time = lin_reg_predict(testing, slope, constant)
accuracy = score(lin_predict, actual_test)
lin_reg_time = time.time() - lin_run + split_time
print "The R^2 value of our linear regression model is " + str(accuracy[0])
print "The run time for linear regression is " + str(lin_reg_time) + " seconds"


#Execute using SKLearn
#Using SkLearn to split apart groups
start_time = time.time()
train, test = sk_split(dataset_1_full, train_size = 0.7)
k=3

#Create matrices to be used in SKlearn functions
train_x = train.as_matrix(['x'])
train_y = train.as_matrix(['y'])
test_x = test.as_matrix(['x'])
test_y = test.as_matrix(['y'])
point1_time = time.time()

#Running SKlearn KNN 
KNNstart_time = time.time()
neighbors = KNN(n_neighbors=k)
neighbors.fit(train_x, train_y)
predicted_y = neighbors.predict(test_x)
r = neighbors.score(test_x, test_y)
endtime = time.time()
totaltime = (point1_time - start_time) + (endtime - KNNstart_time)

print "The R^2 value of the SKLearn KNN model is " + str(r)
print "The SKLearn KNN model takes " + str(totaltime) + " seconds"

#Running sklearn Linear regression 
LinReg_time = time.time()
regression = Lin_Reg()
regression.fit(train_x, train_y)
predicted_y = regression.predict(test_x)
r = regression.score(test_x, test_y)
endtime = time.time()
totaltime = (point1_time - start_time) + (endtime - LinReg_time)

print "The R^2 value of the SKlearn linear regression model is " + str(r)
<<<<<<< HEAD
print "The SKlearn linear regression takes " + str(totaltime) + " seconds".


##Problem 2: Imputation
#input: missing_df (nx2 dataframe, some rows have missing y-values), 
#       full_df (nx2 dataframe, all x-values have correct y-values), 
#       no_y_ind (indices of missing values in missing_df), 
#       with_y_ind (indices of non-missing values in missing_df), 
#       k (integer)
#output: predicted_df (nx2 dataframe, first column is x-vals from missing_df, second column is predicted y-vals), 
#        r (float)
def fill_knn(missing_df, full_df, no_y_ind, with_y_ind, k):
    #preparing data in array form
    
    #training data
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1)) #make x_train array into a 2D array
    y_train = missing_df.loc[with_y_ind, 'y'].values
    
    #testing data
    x_test = missing_df.loc[no_y_ind, 'x']
    x_test = x_test.values.reshape((len(no_y_ind), 1)) #make x_test array into a 2D array
    y_test = full_df.loc[no_y_ind, 'y'].values
    
    #fit knn model
    neighbors = KNN(n_neighbors=k)
    neighbors.fit(x_train, y_train)
    
    #predict y-values
    predicted_y = neighbors.predict(x_test)
    
    #score predictions
    r = neighbors.score(x_test, y_test)
    
    #fill in missing y-values
    predicted_dataframe = missing_df.copy()
    predicted_dataframe.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)
    return predicted_dataframe, r


#input: missing_df (nx2 dataframe, some rows have missing y-values), 
#       full_df (nx2 dataframe, all x-values have correct y-values), 
#       no_y_ind (indices of missing values in missing_df), 
#       with_y_ind (indices of non-missing values in missing_df), 
#       k (integer)
#output: predicted_df (nx2 dataframe, first column is x-vals from missing_df, second column is predicted y-vals), 
#        r (float)
def fill_lin_reg(missing_df, full_df, no_y_ind, with_y_ind):

    #training data
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1)) #make x_train array into a 2D array
    y_train = missing_df.loc[with_y_ind, 'y'].values
    
    #testing data
    x_test = missing_df.loc[no_y_ind, 'x'].values
    x_test = x_test.reshape((len(no_y_ind), 1)) #make x_test array into a 2D array
    y_test = full_df.loc[no_y_ind, 'y'].values
    
    #fit linear model
    regression = Lin_Reg()
    regression.fit(x_train, y_train)
    
    #predict y-values
    predicted_y = regression.predict(x_test)
    
    #score predictions
    r = regression.score(x_test, y_test)
    
    #fill in missing y-values
    predicted_dataframe = missing_df.copy()
    predicted_dataframe.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)
    
    return predicted_dataframe, r



#input: ax1 (axes), ax2 (axes), 
#       predicted_knn (nx2 dataframe with predicted vals), r_knn (float),
#       predicted_lin (nx2 dataframe with predicted vals), r_lin (float), 
#       k (integer),
#       no_y_ind (indices of rows with missing y-values),
#       with_y_ind (indices of rows with no missing y-values)
#output: ax1 (axes), ax2 (axes)
#Create a function to plot data according to KNN and Lin Reg Modeling 
def plot_missing(ax1, ax2, predicted_knn, r_knn, predicted_lin, r_lin, k, no_y_ind, with_y_ind):
    
    #Plot the known values; color blue
    knn_known = ax1.scatter(predicted_knn.loc[with_y_ind]['x'].values, 
                predicted_knn.loc[with_y_ind]['y'].values, 
                color='blue')
    
    #Plot the predicted values; color red
    knn_predicted = ax1.scatter(predicted_knn.loc[no_y_ind]['x'].values, 
                predicted_knn.loc[no_y_ind]['y'].values, 
                color='red')
    
    #Prettify the graph by adding labels, R value, and a legend
    ax1.set_title('Data Set ' + str(i + 1) + ' KNN, R^2:' + str(r_knn) + ', k = ' +str(k))
    ax1.set_xlabel('X-values')
    ax1.set_ylabel('Y-values')
    ax1.legend((knn_known, knn_predicted),
               ("Known", "Predicted"),
               scatterpoints = 1,
               loc='upper left')
    
    #Plot the known values on the second graph; color blue
    lin_known = ax2.scatter(predicted_lin.loc[with_y_ind]['x'].values, 
                predicted_lin.loc[with_y_ind]['y'].values,
                color='blue')
    
    #Plot the predicted values using linear regression; color green
    lin_predicted = ax2.scatter(predicted_lin.loc[no_y_ind]['x'].values, 
                predicted_lin.loc[no_y_ind]['y'].values, 
                color='green')
    
    #Prettify the graph by adding labels, R value, and a legend
    ax2.set_title('Data Set ' + str(i + 1) + ' Linear Regression, R^2:' + str(r_lin))
    ax2.set_xlabel('X-Values')
    ax2.set_ylabel('Y-Values')
    ax2.legend((lin_known, lin_predicted),
               ("Known", "Predicted"),
               scatterpoints = 1,
               loc='upper left')
      
    
    return ax1, ax2


#Create a function that will extract the missing and filled indices of the missing dataframes and plot the scatter plot
def plot_fill (missing_df, full_df):
    
    #Save the indices of missing and filled data, which is denoted by whether the space is null or not
    fig,(ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    no_y_ind = missing_df[missing_df['y'].isnull()].index
    with_y_ind = missing_df[missing_df['y'].notnull()].index
    
    #Use KNN modeling to predict missing data and score it and save within predicted_knn and r_knn
    predicted_knn, r_knn = fill_knn(missing_df, 
                                    full_df, 
                                    no_y_ind, 
                                    with_y_ind, 
                                    k)
    
    #Use linear regression modeling to predict missing data and score it and save within predicted_lin and r_lin
    predicted_lin, r_lin = fill_lin_reg(missing_df, 
                                        full_df, 
                                        no_y_ind, 
                                        with_y_ind)
    
    #Create the two graphs using the plot missing function and inputting output from plot_missing
    ax1, ax2 = plot_missing(ax1, 
                            ax2, 
                            predicted_knn, r_knn,
                            predicted_lin, r_lin,
                            k,
                            no_y_ind, 
                            with_y_ind)
    
    plt.show()


#number of neighbors
k=10

#Read all six dataframes and store into a list of dataframes as touples
dataset_1_missing = pd.read_csv('dataset_1_missing.txt')
dataset_1_full = pd.read_csv('dataset_1_full.txt')
dataset_2_missing = pd.read_csv('dataset_2_missing.txt')
dataset_2_full = pd.read_csv('dataset_2_full.txt')
dataset_3_missing = pd.read_csv('dataset_3_missing.txt')
dataset_3_full = pd.read_csv('dataset_3_full.txt')
dataset_4_missing = pd.read_csv('dataset_4_missing.txt')
dataset_4_full = pd.read_csv('dataset_4_full.txt')
dataset_5_missing = pd.read_csv('dataset_5_missing.txt')
dataset_5_full = pd.read_csv('dataset_5_full.txt')
dataset_6_missing = pd.read_csv('dataset_6_missing.txt')
dataset_6_full = pd.read_csv('dataset_6_full.txt')
dataframe_list = [(dataset_1_missing, dataset_1_full), (dataset_2_missing, dataset_2_full), (dataset_3_missing, dataset_3_full), (dataset_4_missing, dataset_4_full), (dataset_5_missing, dataset_5_full), (dataset_6_missing, dataset_6_full)]

#Loop through all dataframes and plot
for i in xrange(len(dataframe_list)):
    plot_fill(dataframe_list[i][0], dataframe_list[i][1])




###Find the role of K on KNN through numerical analysis and data visualization
#Numerical analysis of K on R^2
#Create two lists to store a k value and its respectful R^2 values
k_list = []
r2_list = []
missing_y_size = dataset_1_missing[dataset_1_missing['y'].notnull()].index.shape[0]

#Loop through each k value, and calculate the r^2 value
for i in range(1,missing_y_size):
    no_y_ind = dataset_1_missing[dataset_1_missing['y'].isnull()].index
    with_y_ind = dataset_1_missing[dataset_1_missing['y'].notnull()].index
    predicted_knn, r2 = fill_knn(dataset_1_missing, dataset_1_full, no_y_ind, with_y_ind, i)
    k_list.append(round(i,3))
    r2_list.append(round(r2, 3))
    print "For a k value of "+ str(int(i)) + ", the R^2 coefficient is " + str("%.3f" %r2)


#Now, we will plot 4 scatter plots: one where it is k vs. R^2, and three scatter plots showing predicted and known 
#Plot the first scatter plot of K values against R^2 coefficients
plt.scatter(k_list, r2_list)
plt.xlabel('k-values')
plt.ylabel('R^2 coefficients')
plt.suptitle('Impact of K on Effectiveness of KNN Modeling')
plt.show()


#Plot three scatter plots of missing data 1 and its KNN predictions using a k-value of 3, 30, and 300
fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))

#Run KNN model with k = 3, then prettify
predicted_knn, r2 = fill_knn(dataset_1_missing, dataset_1_full, no_y_ind, with_y_ind, 3)
predicted = ax1.scatter(predicted_knn.loc[no_y_ind]['x'].values, 
                predicted_knn.loc[no_y_ind]['y'].values, 
                color='red')

known = ax1.scatter(predicted_knn.loc[with_y_ind]['x'].values, 
                 predicted_knn.loc[with_y_ind]['y'].values, 
                 color='blue')

ax1.set_xlabel('X-Values')
ax1.set_ylabel('Y-Values')
ax1.set_title('K = 3, r^2 = ' + str(r2))
ax1.legend((known, predicted),
           ("Known", "Predicted"),
          scatterpoints = 1,
          loc= 'lower right')


#Run KNN model with k = 30, then prettify
predicted_knn, r2 = fill_knn(dataset_1_missing, dataset_1_full, no_y_ind, with_y_ind, 30)
predicted = ax2.scatter(predicted_knn.loc[no_y_ind]['x'].values, 
                predicted_knn.loc[no_y_ind]['y'].values, 
                color='red')

known = ax2.scatter(predicted_knn.loc[with_y_ind]['x'].values, 
                 predicted_knn.loc[with_y_ind]['y'].values, 
                 color='blue')

ax2.set_xlabel('X-Values')
ax2.set_ylabel('Y-Values')
ax2.set_title('K = 30, r^2 = ' + str(r2))
ax2.legend((known, predicted),
           ("Known", "Predicted"),
          scatterpoints = 1,
          loc= 'lower right')


#Run KNN model with k = 300, then prettify
predicted_knn, r2 = fill_knn(dataset_1_missing, dataset_1_full, no_y_ind, with_y_ind, 300)
predicted = ax3.scatter(predicted_knn.loc[no_y_ind]['x'].values, 
                predicted_knn.loc[no_y_ind]['y'].values, 
                color='red')


known = ax3.scatter(predicted_knn.loc[with_y_ind]['x'].values, 
                 predicted_knn.loc[with_y_ind]['y'].values, 
                 color='blue')

ax3.set_xlabel('X-Values')
ax2.set_ylabel('Y-Values')
ax3.set_title('K = 30, r^2 = ' + str(r2))
ax3.legend((known, predicted),
           ("Known", "Predicted"),
          scatterpoints = 1,
          loc= 'lower right')
plt.show()


###Problem 3
#Plot the linear fit, plot of the residuals, and a histogram of residuals for three linear fits for a dataset: two arbitrary ones and the lin reg

#Define a function that plots the three graphs listed above
#Inputs: an nx2 array, slope, and constant
def residuals(data, slope, constant):
    fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    
    y_predicted = data['x']*slope + constant
    
    known = ax1.scatter(data['x'],data['y'], color='green')
    predictions = ax1.scatter(data['x'],y_predicted)
    ax1.set_title('Scatter for slope ' + str("%.2f" %slope) + ' ,constant ' +str("%.2f" %constant))
    ax1.set_xlabel('X-Values')
    ax1.set_ylabel('Y-Values')
    ax1.legend((known, predictions), 
              ("Known", "Predicted"),
              scatterpoints = 1,
              loc = 'upper left')
    
    res = ax2.scatter(data['x'], data['y'] - y_predicted, color='green')
    predictions = ax2.scatter(data['x'],y_predicted)
    ax2.set_title('Residuals for slope ' + str("%.2f" %slope) + ' ,constant ' +str("%.2f" %constant))
    ax2.set_xlabel('X-Values')
    ax2.set_ylabel('Residuals')
    ax2.legend((res, predictions), 
              ("Residuals", "Predicted"),
              scatterpoints = 1,
              loc = 'upper left')
    
    ax3.hist(data['y'] - (y_predicted),
             50,
             normed=1,
             alpha=.75,
             color='green'
            )
    ax3.set_title('Hist. of residuals')
    
    
    #Calculate R^2 by creating dataframes and running score
    predicted = pd.DataFrame(y_predicted)
    predicted.columns = ['Predicted']
    x_val = data.iloc[:,0]
    x_val = pd.DataFrame(x_val)
    final_predictions = pd.concat([x_val, predicted], axis = 1)
    R2, time = score(final_predictions, data)

    plt.show()
    print 'The R^2 coefficient for the regression is ' + str(R2)
    return R2

#Use the function above to plot for the two arbitrary fits
residuals(dataset_1_full, .4, .2)
residuals(dataset_1_full, .4, .4)
slope, constant, z = linear_reg_fit(dataset_1_full)

#Run lin regression and then use resulting slope and constant to fit
residuals(dataset_1_full, slope, constant)




