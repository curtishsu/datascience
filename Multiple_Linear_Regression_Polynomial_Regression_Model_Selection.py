### Multiple Linear Regression, Polynomial Regression, and Model Selection

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy as sp
import statsmodels.api as sm
%matplotlib inline


###Problem 1
#Implment two functions: one that fits a multiple linear regression and then uses it to predict
#Implment a score function
#Use the functions to predict automobile data

### Functions for fitting and evaluating multiple linear regression
### Functions for fitting and evaluating multiple linear regression

#--------  multiple_linear_regression_fit
# A function for fitting a multiple linear regression
# Fitted model: f(x) = x.w + c
# Input: 
#      x_train (n x d array of predictors in training data)
#      y_train (n x 1 array of response variable vals in training data)
# Return: 
#      w (d x 1 array of coefficients) 
#      c (float representing intercept)

def multiple_linear_regression_fit(x_train, y_train):
    
    # Append a column of one's to x
    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
    
    x_train = np.concatenate((x_train, ones_col), axis=1)
    
    # Compute transpose of x
    x_transpose = np.transpose(x_train)
    
    # Compute coefficients: w = inv(x^T * x) x^T * y
    # Compute intermediate term: inv(x^T * x)
    # Note: We have to take pseudo-inverse (pinv), just in case x^T * x is not invertible 
    x_t_x_inv = np.linalg.pinv(np.dot(x_transpose, x_train))
    
    # Compute w: inter_term * x^T * y 
    w = np.dot(np.dot(x_t_x_inv, x_transpose), y_train)
    
    # Obtain intercept: 'c' (last index)
    c = w[-1]
    
    return w[:-1], c


#--------  multiple_linear_regression_score
# A function for evaluating R^2 score and MSE 
# of the linear regression model on a data set
# Input: 
#      w (d x 1 array of coefficients)
#      c (float representing intercept)
#      x_test (n x d array of predictors in testing data)
#      y_test (n x 1 array of response variable vals in testing data)
# Return: 
#      r_squared (float) 
#      y_pred (n x 1 array of predicted y-vals)

def multiple_linear_regression_score(w, c, x_test, y_test):        
    # Compute predicted labels
    y_pred = np.dot(x_test, w) + c
    
    # Evaluate sqaured error, against target labels
    # sq_error = \sum_i (y[i] - y_pred[i])^2
    sq_error = np.sum(np.square(y_test - y_pred))
    
    # Evaluate squared error for a predicting the mean value, against target labels
    # variance = \sum_i (y[i] - y_mean)^2
    y_mean = np.mean(y_test)
    y_variance = np.sum(np.square(y_test - y_mean))
    
    # Evaluate R^2 score value
    r_squared = 1 - sq_error / y_variance

    return r_squared, y_pred


# Load train and test data sets
data_train = np.loadtxt('dataset_1_train.txt', delimiter=',', skiprows=1)
data_test = np.loadtxt('dataset_1_test.txt', delimiter=',', skiprows=1)

# Split predictors from response
# Training
y_train = data_train[:, -1]
x_train = data_train[:, :-1]

# Testing
y_test = data_test[:, -1]
x_test = data_test[:, :-1]

# Fit multiple linear regression model
w, c = multiple_linear_regression_fit(x_train, y_train)

# Evaluate model
r_squared, _ = multiple_linear_regression_score(w, c, x_test, y_test)

print 'R^2 score on test set:', r_squared 

#Compute confidence intervals for the model parameters by creating 200 random subsamples of the data set of size 100
#Use the function to fit a multiple linear regression model to each subsample. 
# each coefficient on the predictor variables: plot a histogram of the values obtained across the subsamples
#Calculate the confidence interval for the coefficients at a confidence level of 95%. 
# Load train set
data = np.loadtxt("dataset_2.txt", delimiter=',', skiprows = 1)

# Size of data set, and subsample (10%)
x = data[:, :-1]
print type(data)
print type(x)
y = data[:, -1]

# Record size of the data set
n = x.shape[0]
d = x.shape[1]
subsample_size = 100

# No. of subsamples
num_samples = 200
    
### Linear regression with all 5 predictors

# Create a n x d array to store coefficients for 100 subsamples
coefs_multiple = np.zeros((num_samples, d))

print 'Linear regression with all predictors'

# Repeat for 200 subsamples
for i in range(num_samples):
    # Generate a random subsample of 50 data points
    perm = np.random.permutation(n) # Generate a list of indices 0 to n and permute it
    x_subsample = x[perm[:subsample_size], :] # Get x-vals for the first 50 indices in permuted list
    
    y_subsample = y[perm[:subsample_size]] # Get y-vals for the first 50 indices in permuted list

    # Fit linear regression model on subsample
    w, c = multiple_linear_regression_fit(x_subsample, y_subsample)
    # Store the coefficient for the model we obtain
    coefs_multiple[i, :] = w

# Plot histogram of coefficients, and report their confidence intervals 
fig, axes = plt.subplots(1, d, figsize=(20, 3))

# Repeat for each coefficient
for j in range(d):
    # Compute mean for the j-th coefficent from subsamples
    coef_j_mean = np.mean(coefs_multiple[:, j])
    
    # Compute confidence interval at 95% confidence level (use formula!)
    conf_int_left = np.percentile(coefs_multiple[:, j], 2.5)
    conf_int_right = np.percentile(coefs_multiple[:, j], 97.5)
    
    #Compute the spread
    coef_var = np.std(coefs_multiple[:, j])
    
    #coef_var = np.sum(np.square(coefs_multiple[:,j] - coef_j_mean))
    #y_variance = np.sum(np.square(y_test - y_mean))
    # Plot histogram of coefficient values
    axes[j].hist(coefs_multiple[:, j], alpha=0.5)

    # Plot vertical lines at mean and left, right extremes of confidence interval
    axes[j].axvline(x = coef_j_mean, linewidth=3)
    axes[j].axvline(x = conf_int_left, linewidth=1, c='r')
    axes[j].axvline(x = conf_int_right, linewidth=1, c='r')
    
    # Set plot labels
    axes[j].set_title('95% Con. Interval: [' + str(round(conf_int_left, 4)) 
                      + ', ' 
                      + str(round(conf_int_right, 4)) + ']\n Spread (Std. Dev): ' + str(coef_var))
    axes[j].set_xlabel('Predictor ' + str(j + 1))
    axes[j].set_ylabel('Frequency')
plt.show()

#Calculate the 95% confidence interval using the entire data set
# Add column of ones to x matrix
x = sm.add_constant(x)

# Create model for linear regression
model = sm.OLS(y, x)
# Fit model
fitted_model = model.fit()
# The confidence intervals for our five coefficients are contained in the last five
# rows of the fitted_model.conf_int() array
conf_int = fitted_model.conf_int()[1:, :]

#Loop through the predictors to find the 
for j in range(d):
    print 'the confidence interval for the', j + 1, 'th coefficient: [', conf_int[j][0], ',', conf_int[j][1], ']'




###Problem 2
#Implement 3 functions: polynomial fit, polynomial predict, and score
#Using provided data set, predict with a degree of 3, 5, 10, and 25 and display predictions in plot graph

#Take an x train, y train, and degrees of polynomial as arguments
#Returns an array of the beta coefficients and a constant

data_3 = pd.read_csv("dataset_3.txt")
x_train = np.array(data_3.iloc[:,0])
y_train = np.array(data_3.iloc[:,1])

def polynomial_regression_fit(x_train, y_train, degree):
    
    #Get number of rows
    n = np.size(y_train)
    x_poly_d = np.zeros([n, degree])
    
    #Loop through for each degree and raise it to a power
    for d in xrange(1, degree + 1):
        x_poly_d[:, d - 1] = np.power(x_train, d)
    
    #Use coefficients to fit the model
    model = multiple_linear_regression_fit(x_poly_d, y_train)
    
    return model

#Takes paramters returns by regression_fit (touple of coefficients and constant), degree of polynomial, and x test values
#Returns y predictions
def polynomial_regression_predict (params, degree, x_test):
    n = x_test.shape[0]
    x_poly = np.zeros([n, degree])
    for d in xrange(1, degree + 1):
        x_poly[:, d - 1] = np.power(x_test, d)
    
    #Add a column of ones to beginning of x values
    ones_col = np.ones((n, 1))
    x_values = np.concatenate((ones_col,x_poly,), axis=1)
    
    #Create single column of beta coefficients
    b = np.insert(params[0], 0, params[1])
    betas = np.reshape(b, (-1, degree + 1))

    y_pred = np.dot(betas, x_values.T)
    y_pred = y_pred
    
    return y_pred

#Takes array of predicted y values and true y values as arguments
#returns R^2 score for the model on the test set and sum of squared errors
def polynomial_regression_score(predicted, y_test):
    
    # Evaluate sqaured error, against target labels
    square_error = np.sum(np.square(y_test - predicted))
    
    # Evaluate squared error for a predicting the mean value, against target labels
    # variance = \sum_i (y[i] - y_mean)^2
    y_mean = np.mean(y_test)
    y_variance = np.sum(np.square(y_test - y_mean))
    
    # Evaluate R^2 score value
    r2 = 1 - square_error / y_variance
    
    return r2, square_error


#Create points, which is an evenly split x variable between the min and max of x_train
points = np.linspace(x_train.min(), x_train.max(), num = 50)

#Predict y values and plot for each degree
poly_fit1 = polynomial_regression_fit(x_train, y_train, 3)
poly_predict1 = polynomial_regression_predict(poly_fit1, 3, points).ravel()

poly_fit2 = polynomial_regression_fit(x_train, y_train, 5)
poly_predict2 = polynomial_regression_predict(poly_fit2, 5, points).ravel()

poly_fit3 = polynomial_regression_fit(x_train, y_train, 10)
poly_predict3 = polynomial_regression_predict(poly_fit3, 10, points).ravel()

poly_fit4 = polynomial_regression_fit(x_train, y_train, 25)
poly_predict4 = polynomial_regression_predict(poly_fit4, 25, points).ravel()


#Plot the four lines and the scatter plot and prettify 
plt.figure(figsize=(18,10))
known = plt.scatter(x_train, y_train, color='blue')
predicted1, = plt.plot(points, poly_predict1, color='red')
predicted2, = plt.plot(points, poly_predict2, color='brown')
predicted3, = plt.plot(points, poly_predict3, color='green')
predicted4, = plt.plot(points, poly_predict4, color='cyan')
plt.legend((known, predicted1, predicted2, predicted3, predicted4),("Known", "Degree 3", "Degree 5", "Degree 10", "Degree 25"))
plt.xlabel('X-values', fontsize = 14)
plt.ylabel('Y-values', fontsize = 14)
plt.suptitle("Predicted values of degrees 3, 5, 10, and 25", fontsize=14)
plt.show()

#These are supplemental plots to help visualize the degree, as well as show the R^2 values
#Create a list of the four degrees; loop through list and plot polynomials
degree_list = [3,5,10,25]
for d in degree_list:
    plot_polynomial(x_train, y_train, d)
plt.show()


###Comparing training and testing data
#Split the data into training and testing data
#Fit polynomials of degree 1 through 15 to the training and testing data
#Plot R^2 and fitted polynomial data model

#Create a function to plot the training and testing data as a visualize analysis
def sub_plot_poly(x_train, y_train, x_test, y_test, degree):
    fig,(ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
    
    #Fit and plot train data
    train_fit = polynomial_regression_fit(x_train, y_train, degree)
    train_predict = polynomial_regression_predict(train_fit, degree, x_train)
    train_r2, square_error = polynomial_regression_score(train_predict, np.array(y_train))
    known_train = ax1.scatter(x_train, y_train, color='blue')
    predicted_train = ax1.scatter(x_train, train_predict, color='red')
    
    #Prettify train data
    ax1.set_title("Train Data" + ", R^2 = " + str(train_r2))
    ax1.set_xlabel('X-Values')
    ax1.set_ylabel('Y-Values')
    ax1.legend((known_train, predicted_train),
              ("Known", "Predicted"))
    
    #Fit and plot test data
    test_predict = polynomial_regression_predict(train_fit, degree, x_test)
    test_r2, square_error = polynomial_regression_score(test_predict, np.array(y_test))
    known_test = ax2.scatter(x_test, y_test, color='blue')
    predicted_test = ax2.scatter(x_test, test_predict, color='green')
    
    #Prettify test data
    ax2.set_title("Test Data" + ", R^2 = " + str(test_r2))
    ax2.set_xlabel('X-Values')
    ax2.set_ylabel('Y-Values')
    ax2.legend((known_test, predicted_test),
              ("Known", "Predicted"))
    
    plt.show()
    return train_r2, test_r2

#Find the midpoint of the length of the data set and split data set 3 into training andtesting
data_3_length = data_3.shape
middle = data_3_length[0]/2
training = data_3.ix[0:middle - 1,:]
x_train = training['x']
y_train = training[' y']

testing = data_3.ix[middle: data_3_length[0],:]
x_test = testing['x']
y_test = testing[' y']

#Create lists of r2 values and degree
train_r2_list = []
test_r2_list = []
degree_list = []

#Plot both training and testing data with polynomials of degree 1- 15
for i in xrange(1,16):
    print "For polynomial degree of " + str(i)
    train_r2, test_r2 = sub_plot_poly(x_train, y_train, x_test, y_test, i)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    degree_list.append(i)


#Plot a scatter plot of the R^2 values on y_axis by the degree on the x axis
plt.figure(figsize=(10,5))
train, = plt.plot(degree_list, train_r2_list, color='blue')
test, = plt.plot(degree_list, test_r2_list, color='red')

#Find the max $R^2$ values for both the train and test data
train_max = train_r2_list.index(max(train_r2_list)) + 1
test_max = test_r2_list.index(max(test_r2_list)) + 1

#Prettify
plt.suptitle('$R^2$ Values of Train and Test Data')
plt.ylabel('$R^2$')
plt.xlabel('X-Values')
plt.legend((train,test), ("Train", "Test"), loc='lower right')
plt.show()

print "According to the R^2 values, the best degree fit for train data is " +str(train_max) + " and the best fit for the test data is " + str(test_max)


###Problem 3: Model Selection
#For each model compute the AIC and BIC and plot as a function of degrees of polynomials


#Takes in a n x 2 matrix with 'x' and 'y' and a degree for the polynomial
#Returns the AIC and BIC value
def model_select(training, degree):
    
    #Extract the x column, y column, and the size of the array
    x_train = np.array(training.iloc[:,0])
    y_train = np.array(training.iloc[:,1])
    n = x_train.shape[0]
    
    #Fit the training data
    train_fit = polynomial_regression_fit(x_train, y_train, degree)
    train_predict = polynomial_regression_predict(train_fit, degree, x_train)
    
    #Calculate RSS, AIC, and BIC
    train_rss = np.sum((y_train - train_predict)**2)
    train_aic = n*np.log(train_rss/n) + 2*degree
    train_bic = n*np.log(train_rss/n) + np.log(n)*degree
    
    return train_aic, train_bic


#Create lists and calculate AIC and BIC for polynomials of degree 1 - 15
aic_list = []
bic_list = []
degrees = []
for i in xrange(1,16):
    ic_values = model_select(training, i)
    aic_list.append(ic_values[0])
    bic_list.append(ic_values[1])
    degrees.append(i)

#Prettify the plot and
aic, = plt.plot(degrees, aic_list, color='red')
bic, = plt.plot(degrees, bic_list, color='blue')
plt.suptitle('AIC/BIC vs. Degrees')
plt.xlabel("Degrees of Polynomial")
plt.ylabel("AI/BIC Value")
plt.legend((aic,bic), ("AIC", "BIC"))
plt.show()

#Find the minimum values of each critereon to find the best fit
aic_min = aic_list.index(min(aic_list)) + 1
bic_min = bic_list.index(min(bic_list)) + 1
print "The best polynomial fit according to AIC is " + str(aic_min) + " degrees because it has the lowest AIC value"
print "The best polynomial fit according to BIC is " + str(bic_min) + " degrees because it has the lowest BIC value"




###Application to NY taxi cab density estimation
#Using the data of taxi cabs in New York, find the best model

data = pd.read_csv('green_tripdata_2015-01.csv', skiprows = [0])

#Create a function that intakes a string of time (hh:mm) and converts it to minutes
def hour_min(time):
    remove = time.split(":")
    time_int = []
    for i in remove:
        time_int.append(int(i))
    minutes = time_int[0] * 60 + time_int[1]
    return minutes


#Splice up the cab data set so that there is an x column(time of day) and y column(number of passengers)
#Create train data by taking only .7 of the dataset
times = np.array(data.iloc[:,1])
pickuptimes = []

#Loop through each element of times to take only the (hh:mm) component
for time in times:
    pickuptime = hour_min(time[11:16])
    pickuptimes.append(pickuptime)
pickuptimes_train = pickuptimes[:int(.7*len(pickuptimes))]

#Extract the frequency of each pick up time to find the density and save as a dataframe
cab_freq = pd.DataFrame(sp.stats.itemfreq(pickuptimes_train))

#Calculate the AIC and BIC of polynomials of degree 1 - 50 and find the index of the lowest AIC/BIC value
cab_aic = []
cab_bic = []
cab_degree = []

for i in xrange(1,51):
    aic, bic = model_select(cab_freq, i)
    cab_aic.append(aic)
    cab_bic.append(bic)
    cab_degree.append(i)
    aic_min_index = cab_aic.index(min(cab_aic)) + 1
    bic_min_index = cab_bic.index(min(cab_bic)) + 1


#Plot the AIC and BIC in respect to degrees of function
aic, = plt.plot(cab_degree, cab_aic, color='red')
bic, = plt.plot(cab_degree, cab_bic, color ='blue')
plt.suptitle('IC Values for Green Cab Data')
plt.xlabel('Degrees of Polynomial')
plt.ylabel('AIC/BIC Values')
plt.legend((aic,bic), ("AIC", "BIC"))
plt.show()
print "According to AIC, the best fit model is a polynomial of degree " + str(aic_min_index)
print "According to BIC, the best fit model is a polynomial of degree " + str(bic_min_index)

def plot_polynomial(x_train, y_train, degree):
    #Plot the known points
    plt.figure(figsize=(10,5))
    known = plt.scatter(x_train, y_train, color='blue')
    
    #Predict y values and plot
    poly_fit = polynomial_regression_fit(x_train, y_train, degree)
    poly_predict = polynomial_regression_predict(poly_fit, degree, x_train).ravel()
    r2, rme = polynomial_regression_score(poly_predict, y_train)
    predicted = plt.scatter(x_train, poly_predict , color='red')
    
    #Prettify
    plt.suptitle('Degrees: ' + str(degree) + '; $R^2$ = ' + str(r2))
    plt.xlabel('X-Values')
    plt.ylabel('Y-Values')
    plt.legend((known, predicted), ("Known", "Predicted"))
    plt.show()


#Make predictions using AIC's and BIC's best model; degree of 47 and 32
#This scatter plot helps visualize the shape of pickup density
#The x-axis is time in minutes and y-axis is pick up data
#Scale to improve picture
cab_time = cab_freq.ix[:,0]/1440
cab_pickups = cab_freq.ix[:,1]/100
plot_polynomial(cab_time, cab_pickups, 47)
plot_polynomial(cab_time, cab_pickups, 32)


#Create a function that intakes a time and predicts the number of pickups demanded according to the
#suggested by both lowest AIC and BIC values
def cab_prediction(time):
    minutes = hour_min(time)
    cab_time = cab_freq.ix[:,0]
    cab_pickups = cab_freq.ix[:,1]
    cab_fit_32 = polynomial_regression_fit(cab_time, cab_pickups, 32)
    pickup_density_pred_32 = polynomial_regression_predict(cab_fit_32, 32, pd.Series(minutes))
    cab_fit_47 = polynomial_regression_fit(cab_time, cab_pickups, 47)
    pickup_density_pred_47 = polynomial_regression_predict(cab_fit_47, 47, pd.Series(minutes)) 
    
    return float('%.02f' %pickup_density_pred_32), ('%.02f' %pickup_density_pred_47)

#To test the function, we'll input a time of 12:00,
pred1, pred2 = cab_prediction('12:00')
print "At 12:00 p.m., our model suggests around " + str(pred1) + " and " + str(pred2) + " pickups"