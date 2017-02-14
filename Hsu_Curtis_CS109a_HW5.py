
# coding: utf-8

# # CS 109A/AC 209A/STAT 121A Data Science: Homework 5
# **Harvard University**<br>
# **Fall 2016**<br>
# **Instructors: W. Pan, P. Protopapas, K. Rader**<br>
# **Due Date: ** Wednesday, October 26th, 2016 at 11:59pm

# To submit your assignment, in Vocareum, upload (using the 'Upload' button on your Vocareum Jupyter Dashboard) your solution to Vocareum as a single notebook with following file name format:
# 
# `last_first_CourseNumber_HW4.ipynb`
# 
# where `CourseNumber` is the course in which you're enrolled (CS 109a, Stats 121a, AC 209a). Submit your assignment in Vocareum using the 'Submit' button.
# 
# **Verify your submission by checking your submission status on Vocareum!**
# 
# **Avoid editing your file in Vocareum after uploading. If you need to make a change in a solution. Delete your old solution file from Vocareum and upload a new solution. Click submit only ONCE after verifying that you have uploaded the correct file. The assignment will CLOSE after you click the submit button.**
# 
# Problems on homework assignments are equally weighted. The Challenge Question is required for AC 209A students and optional for all others. Student who complete the Challenge Problem as optional extra credit will receive +0.5% towards your final grade for each correct solution. 

# Import libraries

# In[93]:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score 
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.svm.libsvm import predict_proba
get_ipython().magic(u'matplotlib inline')


# ## Problem 0: Basic Information
# 
# Fill in your basic information. 
# 
# ### Part (a): Your name

# Hsu, Curtis

# ### Part (b): Course Number

# CS 109a

# ### Part (c): Who did you work with?

# Ted Zhu

# **All data sets can be found in the ``datasets`` folder and are in comma separated value (CSV) format**

# ## Problem 1: Image Classification
# 
# In this problem, your task is to classify images of handwritten digits. 
# 
# The data set is provided in the file `dataset_1.txt` and contains 8x8 gray-scale images of hand-written digits, flattened to a 64-length vector. The last column contains the digit. For simplicity, we have only included digits 0, 1 and 3. 
# 
# We want you to build a model that can be given the image of a hand-written digit and correctly classify this digit as 0, 1 or 3.

# ### Part 1(a).  Reduce the data
# 
# Images data are typically high dimensional (the image vector has one feature for every pixel). Thus, to make working with image data more tractible, one might first apply a dimension reduction technique to the data.
# 
# - Explain why PCA is a better choice for dimension reduction in this problem than step-wise variable selection.
# 
# 
# - Choose the smallest possible number of dimensions for PCA that still permits us to perform classification. 
# 
#   (**Hint:** how do we visually verify that subgroups in a dataset are easily classifiable?)
# 
# 
# - Visualize and interpret the principal components. Interpret, also, the corresponding PCA varaiable values.

# In[173]:

#Load the data
data = np.loadtxt('dataset_1.txt', delimiter=',')

#Split into predictor and response
x = data[:, :-1]
y = data[:, -1]

#Print shapes of predictor and response arrays
print 'predictor matrix shape:', x.shape
print 'response array shape:', y.shape


# In[174]:

#Plot a couple of images from the dataset
fig, ax = plt.subplots(2, 3, figsize=(15, 5))

#Plot the 0th image vector
ax[0, 0].imshow(x[0].reshape(8, 8), cmap=plt.cm.gray_r)
ax[0, 0].set_title('0-th image vector')


#Plot the 300th image vector
ax[0, 1].imshow(x[300].reshape(8, 8), cmap=plt.cm.gray_r)
ax[0, 1].set_title('300-th image vector')


#Plot the 400th image vector
ax[0, 2].imshow(x[400].reshape(8, 8), cmap=plt.cm.gray_r)
ax[0, 2].set_title('400-th image vector')


#Plot the 0th image vector, with de-blurring
ax[1, 0].imshow(x[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
ax[1, 0].set_title('0-th image vector, de-blurred')


#Plot the 300th image vector, with de-blurring
ax[1, 1].imshow(x[300].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
ax[1, 1].set_title('300-th image vector, de-blurred')

#Plot the 400th image vector, with de-blurring
ax[1, 2].imshow(x[400].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
ax[1, 2].set_title('400-th image vector, de-blurred')


plt.tight_layout()
plt.show()


# In[175]:

#Firt, experiment with trial and error
#Let's project the data onto some random 2D planes
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

#Project onto axes: 1, 2
x_2d = x[:, [1, 2]]

ax[0].scatter(x_2d[y==0, 0], x_2d[y==0, 1], color='b', label='0')
ax[0].scatter(x_2d[y==1, 0], x_2d[y==1, 1], color='r', label='1')
ax[0].scatter(x_2d[y==3, 0], x_2d[y==3, 1], color='g', label='3')

ax[0].set_xlabel('X_1')
ax[0].set_ylabel('X_2')
ax[0].set_title('Project of Data onto X_1 and X_2')
ax[0].legend()

#Project onto axes: PICK TWO RANDOM PREDICTORS
x_2d_1 = x[:, [4,63]]
ax[1].scatter(x_2d_1[y==0, 0], x_2d_1[y==0, 1], color='b', label='0')
ax[1].scatter(x_2d_1[y==1, 0], x_2d_1[y==1, 1], color='r', label='1')
ax[1].scatter(x_2d_1[y==3, 0], x_2d_1[y==3, 1], color='g', label='3')

ax[1].set_xlabel('X_4')
ax[1].set_ylabel('X_63')
ax[1].set_title('Project of Data onto X_4 and X_63')
ax[1].legend()


#Project onto axes: PICK TWO RANDOM PREDICTORS
x_2d_2 = x[:, [12,43]]
ax[2].scatter(x_2d_2[y==0, 0], x_2d_2[y==0, 1], color='b', label='0')
ax[2].scatter(x_2d_2[y==1, 0], x_2d_2[y==1, 1], color='r', label='1')
ax[2].scatter(x_2d_2[y==3, 0], x_2d_2[y==3, 1], color='g', label='3')

ax[2].set_xlabel('X_12')
ax[2].set_ylabel('X_43')
ax[2].set_title('Project of Data onto X_12 and X_43')
ax[2].legend()


plt.tight_layout()
plt.show()


# In[176]:

#Let's project the data onto some random 2D planes
fig = plt.figure(figsize=(15, 5))


#Project onto axes: 1, 2, 3
x_2d = x[:, [1, 2, 3]]

ax1 = fig.add_subplot(1, 3, 1,  projection='3d')

ax1.scatter(x_2d[y==0, 0], x_2d[y==0, 1], x_2d[y==0, 2], c='b', color='b', label='0')
ax1.scatter(x_2d[y==1, 0], x_2d[y==1, 1], x_2d[y==1, 2], c='r', color='r', label='1')
ax1.scatter(x_2d[y==3, 0], x_2d[y==3, 1], x_2d[y==3, 2], c='g', color='g', label='3')

ax1.set_xlabel('X_1')
ax1.set_ylabel('X_2')
ax1.set_zlabel('X_3')
ax1.set_title('Project of Data onto X_1, X_2, X_3')
ax1.legend(loc='lower left')

#Project onto axes: 10, 20, 30
x_2d = x[:, [10, 20, 30]]

ax2 = fig.add_subplot(1, 3, 2,  projection='3d')

ax2.scatter(x_2d[y==0, 0], x_2d[y==0, 1], x_2d[y==0, 2], c='b', color='b', label='0')
ax2.scatter(x_2d[y==1, 0], x_2d[y==1, 1], x_2d[y==1, 2], c='r', color='r', label='1')
ax2.scatter(x_2d[y==3, 0], x_2d[y==3, 1], x_2d[y==3, 2], c='g', color='g', label='3')

ax2.set_xlabel('X_10')
ax2.set_ylabel('X_20')
ax2.set_zlabel('X_30')
ax2.set_title('Project of Data onto X_10, X_20, X_30')
ax2.legend(loc='lower left')

#Project onto axes: 1, 50, 63
x_2d = x[:, [1, 50, 63]]

ax3 = fig.add_subplot(1, 3, 3,  projection='3d')

ax3.scatter(x_2d[y==0, 0], x_2d[y==0, 1], x_2d[y==0, 2], c='b', color='b', label='0')
ax3.scatter(x_2d[y==1, 0], x_2d[y==1, 1], x_2d[y==1, 2], c='r', color='r', label='1')
ax3.scatter(x_2d[y==3, 0], x_2d[y==3, 1], x_2d[y==3, 2], c='g', color='g', label='3')

ax3.set_xlabel('X_1')
ax3.set_ylabel('X_50')
ax3.set_zlabel('X_63')
ax3.set_title('Project of Data onto X_1, X_50, X_63')
ax3.legend(loc='lower left')

plt.tight_layout()
plt.show()


# In[177]:

#Apply PCA to data and get the top 3 axes of maximum variation
pca = PCA(n_components=3)
pca.fit(x)
#Project to the data onto the three axes
x_reduced = pca.transform(x)

#Visualized our reduced data
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 2, 1,  projection='3d')
ax1.scatter(x_reduced[y==0, 0], x_reduced[y==0, 1], x_reduced[y==0, 2], c='b', color='b', label='0')
ax1.scatter(x_reduced[y==1, 0], x_reduced[y==1, 1], x_reduced[y==1, 2], c='r', color='r', label='1')
ax1.scatter(x_reduced[y==3, 0], x_reduced[y==3, 1], x_reduced[y==3, 2], c='g', color='g', label='3')

ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')
ax1.set_zlabel('Component 3')
ax1.set_title('data projected onto the first 3 PCA components')
ax1.legend()


#Apply PCA to data and get the top 2 axes of maximum variation
pca = PCA(n_components=2)
pca.fit(x)

#Project to the data onto the three axes
x_reduced = pca.transform(x)

#Visualized our reduced data
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(x_reduced[y==0, 0], x_reduced[y==0, 1], c='b', color='b', label='0')
ax2.scatter(x_reduced[y==1, 0], x_reduced[y==1, 1], c='r', color='r', label='1')
ax2.scatter(x_reduced[y==3, 0], x_reduced[y==3, 1], c='g', color='g', label='3')

ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
ax2.set_title('data projected onto the first 2 PCA components')
ax2.legend()

plt.tight_layout()
plt.show()


# In[178]:

#Display the principal components of PCA as digital images
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
# COMPONENT 1
ax[0, 0].imshow(pca.components_[0].reshape(8,8), cmap=plt.cm.gray_r)
ax[0, 0].set_title('Component 1')
 
# COMPONENT 2
ax[0, 1].imshow(pca.components_[1].reshape(8,8), cmap=plt.cm.gray_r)
ax[0, 1].set_title('Component 2')

# COMPONENT 1
ax[1, 0].imshow(pca.components_[0].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
ax[1, 0].set_title('Component 1: de-blurred')

# COMPONENT 2
ax[1, 1].imshow(pca.components_[1].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
ax[1, 1].set_title('Component 2: de-blurred')

plt.tight_layout()
plt.show()


# **Question Explain why PCA is a better choice for dimension reduction in this problem than step-wise variable selection.**
# PCA is a better chocie for dimension reduction because, as seen in the graphs above, step-wise variable reduction projections don't capture the variation in the data and the classes are not sepearted when we project. The lack of separation means we won't have good classification. With PCA, we project axes that capture the maximum variation, so we are able to maintain more variation in the data. 
# 
# By contruction, the first 2 or 3 components of PCA captures the top 2 or 3 directions of maximum variation. Luckily in our case, the top 2 and 3 components also captures the **separation** in the classes! In fact, we realize that using 2 components of the PCA already separates all three classes! That means we can get away with using just two (linear combinations of) predictors as the smallest number of predictors!
# 
# **Question:** Does the directions of maximum variation always correspond to maximum separation of the classes? That is, is projecting our data onto the top components of the PCA always a good idea for classification?
# 
# **Question:** Recall that the components of PCA are **linear combos of our original predictors**. E.g. component 1 might be 
# $$X_1 - 2 X_{10} + 10 X_{63}.$$
# But in our case, our predictors are pixels, thus, each PCA component is a combination of different pixels - that is, each PCA component is a digital image! This is good news! It means that the components of the PCA are potentially interpretable.
# 
# 
# The first component looks like the digit 0 and the second resembles the digit 3! 
# 
# Look at our data projected onto the first two components: 
# 
# 1. nearly all the data points corresponding to 0 are expressed as **a combination of a positive multiple of component 1 and a negative multiple of component 2**;
# 
# 2. nearly all the data points corresponding to 3 are expressed as **a combination of a negative multiple of component 1 and a positive multiple of component 2**;
# 
# 3. Nearly all the data points corresponding to the digit 1 is expressed as **a negative combination of the two components**.
# 

# ### Part 1(b). Build a classifier
# 
# So far, we have only learned models that distinguishes between two classes. Develop and implement a **simple and naive** method of distinguishing between the three digits in our reduced dataset using binary classifiers. 

# In[179]:

###We will split our data to get an accurate test
x_reduce, x_test, y_train, y_test = sk_split(x_reduced, y, test_size = .3)


# In[180]:

###Build a classifier to distinguish between 0 and 1

#Remove all instances of class 3
x_binary = x_reduce[y_train != 3, :]

#Remove all instances of class 3
y_binary = y_train[y_train != 3]

#Fit logistic regression model for 0 vs 1
logistic_01 = LogReg()
logistic_01.fit(x_binary, y_binary)

###Build a classifier to distinguish between 1 and 3

#Remove all instances of class 0
x_binary = x_reduce[y_train != 0, :]

#Remove all instances of class 0
y_binary = y_train[y_train != 0]

#Fit logistic regression model for 1 vs 3
logistic_13 = LogReg()
logistic_13.fit(x_binary, y_binary)

###Build a classifier to distinguish between 0 and 3

#Remove all instances of class 1
x_binary = x_reduce[y_train != 1, :]

#Remove all instances of class 1
y_binary = y_train[y_train != 1]

#Fit logistic regression model for 0 vs 3
logistic_03 = LogReg()
logistic_03.fit(x_binary, y_binary)

#Predict a label for our dataset using each binary classifier
y_pred_01 = logistic_01.predict(x_test)
y_pred_13 = logistic_13.predict(x_test)
y_pred_03 = logistic_03.predict(x_test)


#Now, for each image, we have THREE predictions!
#To make a final decision for each image, we just take a majority vote.
n = x_test.shape[0]
y_votes = np.zeros((n, 3))
#print y_pred_01
#Votes for 0
y_votes[y_pred_01 == 0, 0] += 1
y_votes[y_pred_03 == 0, 0] += 1

#Votes for 1
y_votes[y_pred_01 == 1, 1] += 1
y_votes[y_pred_13 == 1, 1] += 1

#Votes for 3
y_votes[y_pred_03 == 3, 2] += 1
y_votes[y_pred_13 == 3, 2] += 1

#For each image, label it with the class that get the most votes
y_pred = y_votes.argmax(axis = 1)

#Relabel class 2 as class 3
y_pred[y_pred == 2] = 3

#Accuracy of our predictions
print 'Accuracy of combined model:', np.mean(y_test == y_pred)


# ### Part 1(c). Build a better one
# Asses the quality of your classifier.
# 
# 
# - What is the fit (in terms of accuracy or R^2) of your model on the reduced dataset? Visually assess the quality of your classifier by plotting decision surfaces along with the data. Why is visualization of the decision surfaces useful? What does this visualization tell you that a numberical score (like accuracy or R^2) cannot?
# 
# 
# - What are the draw backs of your approach to multi-class classification? What aspects of your method is contributing to these draw backs, i.e. why does it fail when it does? 
# 
#   (**Hint:** make use your analysis in the above; think about what happens when we have to classify 10 classes, 100 classes)
#  
#  
# - Describe a possibly better alternative for fitting a multi-class model. Specifically address why you expect the alternative model to outperform your model.
# 
#   (**Hint:** How does ``sklearn``'s Logistic regression module handle multiclass classification?).

# There many ways to verify the meaningfulness of our metrics (like R^2 and accuracy rate), in this case, since the reduced data is low dimensional, we can visually check the "goodness" of our classifier. I.e. we can plot our data and visualize the decision boundaries (the lines on which logistic regression rely to separate one class from another).

# In[10]:

#--------  fit_and_plot_model
# A function to fit a binary LogReg model and visualize it
# Input: 
#      model (LogReg model)
#      ax (axes object for plotting)
#      legend_label (legend label for the plot)

def plot_model(model, ax, legend_label, color):
    #Get the coefficients from logistic regression model
    coef = model.coef_[0]
    intercept = model.intercept_
    
    #Find the max and min horizontal values of our data
    x_0 = np.min(x_reduced[:, 0])
    x_1 = np.max(x_reduced[:, 0])
        
    #Plug int the max and min horizontal values of our data into the equation
    #of the line defined by the coefficients
    y_0 = (-intercept - coef[0] * x_0) / coef[1]
    y_1 = (-intercept - coef[0] * x_1) / coef[1]

    #Plot a line through the pair of points we found above
    ax.plot([x_0, x_1], [y_0, y_1], label=legend_label, color=color)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

#Scatter plot of our data
ax.scatter(x_reduced[y==0, 0], x_reduced[y==0, 1], color='b', label='0')
ax.scatter(x_reduced[y==1, 0], x_reduced[y==1, 1], color='r', label='1')
ax.scatter(x_reduced[y==3, 0], x_reduced[y==3, 1], color='g', label='3')

#Plot decision boundaries for 0 vs 1
plot_model(logistic_01, ax, '0 vs 1', 'magenta')
#Plot decision boundaries for 1 vs 3
plot_model(logistic_13, ax, '1 vs 3', 'cyan')
#Plot decision boundaries for 0 vs 3
plot_model(logistic_03, ax, '0 vs 3', 'orange')

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_xlim([np.min(x_reduced[:,0]), np.max(x_reduced[:,0])])
ax.set_ylim([np.min(x_reduced[:,1]), np.max(x_reduced[:,1])])
ax.set_title('our reduced data with all three decision boundaries (from binary classifiers)')
ax.legend()
plt.show()


# **Solution** The accuracy of our model is 0.9264. Visualization is helpful because it gives an idea of the entire data set. When it comes to accuracy as a numeric value, it is possible that our model has functional form issues - that is, we use a model that is sufficient, but has a fundamental issue. In this model, perhaps it could be that we don't have the correct number of PCA components, and that could be seen by visualization but not as easily with accuracy values. Furthermore, visualization helps us confirm that our components are successful at separating data. A draw back of our approach to classification is that our voting system allocates equal weight to each 'vote'. That is, if we had 100 classes and there is one class that is only slightly better than many classes and one class that is significantly better to just one less class, then our model will choose the only slightly better class. Another drawback is that there is a possibility of a tie. When there is a tie, we are unable to choose the best classifier. Another issue drawback is that, by using our method, we must compare each class with another class to finally get our answer. Thus we most do n choose 2 combinations where n is the number of classes. This is inefficient. 
# 
# A possibly better alternative to fitting multi-class model is use a multinomial algorithm instead of using one vs. one algorithm. By doing so, we are able to compare the probability of one class relative to all the other classes and figure out our decision boundaries without individually comparing that class with each of the other classes. As a result, we are able to more efficiently and elegantly calculate the probabilities that a data point will fall into a class when there are multiple classes. Therefore, a multinomial algorithm will outperform the algorithm that was implemented above. 

# ## Problem 2. Sentiment Analysis

# In this problem, you will explore how to predict the underlying emotional tone of textual data - this task is called sentiment analysis. 
# 
# You will be using the dataset in the file `dataset_2.txt`. In this dataset, there are 1382 posts containing textual opinions about Ford automobiles, along with labels indicating whether the opinion expressed is positive or negative. 
# 
# Given a new post about an automobile, your goal is to predict if the sentiment expressed in the new post is positive or negative. For this task you should implement a *regularized* logistic regression model.
# 
# Produce a report summarizing your solution to this problem:
# 
# - Your report should address all decisions you made in the "Data Science Process" (from Lectures #0, #1, #2):
# 
#    a. Data collection & cleaning
#    
#    b. Data exploration
#    
#    c. Modeling
#    
#    d. Analysis  
#    
#    e. Visualization and presentation  
# 
# 
# - Your report should be informative and accessible to a **general audience with no assumed formal training in mathematics, statistics or computer science**.
# 
# 
# - The exposition in your report, not including code, visualization and output, should be at least three paragraphs in length (you are free to write more, but you're not required to).
# 
# Structure your presentation and exposition like a professional product that can be submitted to a client and or your supervisor at work.

# In[47]:

#Import the data and vectorize both the X and the Y
df = pd.read_csv('dataset_2.txt', delimiter=',')

vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the'], min_df=5)
corpus_x = df['text'].values
x = vectorizer.fit_transform(corpus_x)
x = x.toarray()

vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the'], min_df=5)
corpus_y = df['class'].values
y = vectorizer.fit_transform(corpus_y)
y = y.toarray()
y = y[:,0]

x_train, x_test, y_train, y_test = sk_split(x, y, test_size=0.3)


# In[48]:

#Explore the data to get things such as the number of positives and negatives of our posts
neg_posts = np.sum(y)
pos_posts = len(y) - neg_posts
print 'There are ' +str(neg_posts) + ' negative posts and ' + str(pos_posts) + ' positive posts'
plt.pie([neg_posts, pos_posts], labels = ['Negative', 'Positive'], autopct= '%1.1f%%')
plt.axis('equal')
plt.show()
print 'We also know that there are ' + str(x.shape[1]) + ' distinct features that we are using'


# In[49]:

#Exploring the data w/ PCA
#Apply PCA to data and get the top 3 axes of maximum variation
#Note: in the write up I explain why PCA doesn't really make sense in this situation
pca = PCA(n_components=3)
pca.fit(x)
#Project to the data onto the three axes
x_reduced = pca.transform(x)

#Visualized our reduced data
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 2, 1,  projection='3d')
ax1.scatter(x_reduced[y==1, 0], x_reduced[y==0, 1], x_reduced[y==0, 2], c='b', color='b', label='Neg')
ax1.scatter(x_reduced[y==0, 0], x_reduced[y==1, 1], x_reduced[y==1, 2], c='r', color='r', label='Pos')

ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')
ax1.set_zlabel('Component 3')
ax1.set_title('data projected onto the first 3 PCA components')
ax1.legend()


#Apply PCA to data and get the top 2 axes of maximum variation
pca = PCA(n_components=2)
pca.fit(x)

#Project to the data onto the three axes
x_reduced = pca.transform(x)

#Visualized our reduced data
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(x_reduced[y==1, 0], x_reduced[y==0, 1], c='b', color='b', label='Neg')
ax2.scatter(x_reduced[y==0, 0], x_reduced[y==1, 1], c='r', color='r', label='Pos')


ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
ax2.set_title('data projected onto the first 2 PCA components')
ax2.legend()

plt.tight_layout()
plt.show()


# In[50]:

#Now that PCA didn't work in classifying, we will try regularizing to find the best model
#Create a function to a run a regression give a regulizing paramter
#That will cross validate
#Return the accuracy score
def cross_val_l1(c, x_train, y_train):
    kf = KFold(x_train.shape[0], n_folds=5, shuffle = True)
    logistic_l1 = LogReg(C=c, penalty = "l1")
    score = cross_val_score(logistic_l1, x_train, y_train, cv =kf)
    avg_score = score.mean()
    return avg_score

def cross_val_l2(c, x_train, y_train):
    kf = KFold(x_train.shape[0], n_folds=5, shuffle = True)
    logistic_l1 = LogReg(C=c, penalty = "l2")
    score = cross_val_score(logistic_l1, x_train, y_train, cv =kf)
    avg_score = score.mean()
    return avg_score


# In[51]:

#Plot the values
#Loop through 15 possible c scores 
accuracy_l1 = []
accuracy_l2 = []
for i in xrange(-7, 8):
    c = 10**i
    acc_l1 = cross_val_l1(c, x_train, y_train)
    accuracy_l1.append(acc_l1)
    acc_l2 = cross_val_l2(c, x_train, y_train)
    accuracy_l2.append(acc_l2)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.semilogx(10.0**np.arange(-7,8), 
            accuracy_l1,
            label = 'Lasso')
ax.semilogx(10.0**np.arange(-7, 8), 
            accuracy_l2, label = 'Ridge')

ax.set_ylim(.4, .9)
ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Paramters')
ax.legend()

plt.show()


# We see in the above graph that ridge regression produces higher accuracy scores than Lasso regularization. Therefore, we would prefer using ridge regression

# In[52]:

l1_max_index = np.argmax(accuracy_l1) -7
l1_r1_max = np.max(accuracy_l1)
l2_max_index = np.argmax(accuracy_l2) -7
l2_r2_max = np.max(accuracy_l2)

print "According to our Lasso regularization, we would prefer a regularization value of 10^" + str(l1_max_index) + " which produces an R^2 value of " + str(l1_r1_max)  
print "According to our Ridge regularization, we would prefer a regularization value of 10^" + str(l2_max_index) + " which produces an R^2 value of " + str(l2_r2_max)  


# In[53]:

#Fit model using our test data
logit_l1 = LogReg(C = 10**l1_max_index, penalty='l1')
logit_l1.fit(x_train, y_train)
acc_l1 = logit_l1.score(x_test, y_test)

logit_l2 = LogReg(C = 10**l2_max_index, penalty='l2')
logit_l2.fit(x_train, y_train)
acc_12 = logit_l2.score(x_test, y_test)
print "The accuracy of our lasso method is " + str(acc_l1)
print "The accuracy of our ridge regression algorithm " + str(acc_12)


# In[54]:

#We are interested in finding the most popular words and seeing if they are neutral
#If they are neutral, then we could remove them, because they just add noise. This creates a better model
x = vectorizer.fit_transform(corpus_x)
x = x.toarray()

#Calculate the frequency of words and get the index and name of the most popular ones
#Because they are all neutral, we remove them
#While Ridge regression does minimize these values, we want to explore if doing this will also improve our model
freq =  sum(x)
count = np.arange(0,x[0].shape[0])
plt.plot(count,freq)
lst_freq = list(freq)
val = list(freq[freq >=2500])
index_val = []
for i in xrange(0,len(val)):
    index_val.append(lst_freq.index(val[i]))
lst_pop_words = []

for i in xrange(0, len(index_val)):
    lst_pop_words.append(vectorizer.get_feature_names()[index_val[i]])
print lst_pop_words


# In[55]:

#Re vectorize, this time including the stop words from above
vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the', 'are', 'as', 'at', 'be', 'but', 'car', 'for', 'ford', 'had', 'has', 'have', 'in', 'is', 'it', 'my', 'not', 'of', 'that', 'this', 'to', 'was','we', 'with', 'you'], min_df=5)
corpus_x = df['text'].values
x = vectorizer.fit_transform(corpus_x)
x = x.toarray()

vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the'], min_df=5)
corpus_y = df['class'].values
y = vectorizer.fit_transform(corpus_y)
y = y.toarray()
y = y[:,0]

x_train, x_test, y_train, y_test = sk_split(x, y, test_size=0.3)


# In[56]:

#Plot the values
#Get a new C score with our new x array
accuracy_l1 = []
accuracy_l2 = []
for i in xrange(-7, 8):
    c = 10**i
    acc_l1 = cross_val_l1(c, x_train, y_train)
    accuracy_l1.append(acc_l1)
    acc_l2 = cross_val_l2(c, x_train, y_train)
    accuracy_l2.append(acc_l2)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.semilogx(10.0**np.arange(-7,8), 
            accuracy_l1,
            label = 'Lasso')
ax.semilogx(10.0**np.arange(-7, 8), 
            accuracy_l2, label = 'Ridge')

ax.set_ylim(.4, .9)
ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Paramters')
ax.legend()

plt.show()


# In[57]:

l1_max_index = np.argmax(accuracy_l1) -7
l1_r1_max = np.max(accuracy_l1)
l2_max_index = np.argmax(accuracy_l2) -7
l2_r2_max = np.max(accuracy_l2)

print "According to our Lasso regularization, we would prefer a regularization value of 10^" + str(l1_max_index) + " which produces an R^2 value of " + str(l1_r1_max)  
print "According to our Ridge regularization, we would prefer a regularization value of 10^" + str(l2_max_index) + " which produces an R^2 value of " + str(l2_r2_max)  


# In[58]:

#Fit model using our test data
logit_l1 = LogReg(C = 10**l1_max_index, penalty='l1')
logit_l1.fit(x_train, y_train)
acc_l1 = logit_l1.score(x_test, y_test)

logit_l2 = LogReg(C = 10**l2_max_index, penalty='l2')
logit_l2.fit(x_train, y_train)
acc_12 = logit_l2.score(x_test, y_test)
print "The accuracy of our lasso method is " + str(acc_l1)
print "The accuracy of our ridge regression algorithm " + str(acc_12)


# In[59]:

#Create a function that will intake a text as a comment
#Output will be an integer that predicts whether something is positive or negative
#Because ridge had a higher test score, we will use that
def predict_sentiment(text):
    vectorizer = CountVectorizer(stop_words=['and', 'or', 'before', 'a', 'an', 'the', 'are', 'as', 'at', 'be', 'but', 'car', 'for', 'ford', 'had', 'has', 'have', 'in', 'is', 'it', 'my', 'not', 'of', 'that', 'this', 'to', 'was','we', 'with', 'you'], min_df=5)

    vectorizer.fit_transform(corpus_x)
    text_transform = vectorizer.transform([text])
    y_pred = logit_l2.predict(text_transform)
    if(y_pred[-1] == 0):
        return "Positive"
    elif(y_pred[-1] ==1):
        return "Negative"
    return False

    


# In[60]:

#Test this out on a few phrases
print predict_sentiment('I love CS109a! It is my favorite class')
print predict_sentiment('Bad would not buy this product ever again')


# ## Report
# 
# ** Exposition **: 
# Our mission for this project is to use past posts about Ford automobiles that have been cateogorized as positive or negative to build a logistic model, that will, intake a new text post and predict whether the post is positive or negative. For **data collection** we have collected 1382 posts about the Ford automobile. Then, they were categorized as positive or negative. In terms of collection, this is enough posts that we can build a sufficient model so collection has been taken care of. To **clean** , we will remove words that show up less than 5% of the documents. The reason we do this is because this will remove uncommon words, many Personal nouns, and misspelled words. To quantify our text posts, we will 'vectorize' them. Vectorizing means that we will create a 'feature' for each unique word, and then store the number of time that word shows up for each post. Then we will vectorize the cateogrization of the posts: that is, we will assign a 0 for positive posts and a 1 for negative posts. Now we have our 'x' matrix, which is the words that were used in a post, and a 'y' matrix, which is whether something is positive or negative, there is quantified, so we can build a logistic model to it. The reason we use a logistic model is because, logistic models will provide us with a binary classifier - a positive or a negative, and that's we're interested in. 
# 
# For **Data Exploration ** We want to first find some basic characteristics about our data set. We notice that our posts are about 50/50 split in terms of positive or negative comments, which we visualized in the pie chart. We also see that, after vectorizing, we are using 5123 features to predict whether a post is positive or negative. As we can see, that's quite a few features. In a best case scenario, we'll be able to reduce this to a fewer number of features. To explore this, we will use Principal Component Analysis, which hopes to combine diferent features to form new ones that capture a significant amount of the variance. After performing this reduction, we visualize the data below. There are two main issues with this approach, however. The first issue is that, as can be seen in the graphs below, there are no distinct groups or classes. This means that this reduction does not allow us to accurately classify a post based off of the classifiers. Another issue with using PCA is that it uses linear combinations of words to form new features. However, when it comes to words, the synergy among words does not allow for combinations to be significant better predictors. For example, if we had one feature be dog and another feature be cat, having the two combine would not be that impactful of a feature. Thus, while PCA does give us an interesting aspect of exploration, we must turn to another method to regularize/reduce our data to form our model.
# 
# **Modeling/Analysis** Before we create our model, we first needed to divide our data into a train and test set. The reason we do this is because we want fit our model only our train without touching our test data to authentically 'test' our test data. To create our model, we need to implement a regularized logistic model. What this means is that, due to the high number of predictors that we had found in the previous section, we need to penalize each beta coefficient to prevent over fitting, which is when our betas describe the error instead of the actual relationship we are looking for. For example, if we had ten predictors and ten data points, then eaach predictor could correspond with each data point instead of actually mapping out the relationship. Regualarization accounts for that. Furthermore, a logistic model is best used when we want to classify things; in this case, we are classifying comments as either positive or negative. 
# 
# In our case, we will be regularizing our logistic model using two types of penalties: l1 and l2. We'll be using the one that gives us the highest accuracy score. For each penalty, we have a regularizing paramter, which can be interpreted as the amount that we are regularizing. To find out how much we should regularize our model, we'll try 16 different values for our paramters and Kfold cross validate. KFold cross validation is essentially a method of dividing up our training into different folds, and then iterating through each fold and treating it like a test set. This way, we get an accurate measure of the accuracy of each model when we change the regularization paramter. This way, we can find the best paramter because it will have the highest accuracy. We can look at the visual to see a graph representing the relationship between our paramter and accuracy. Using that paramter, we constructed a logistic model that we used against our test data. Our best accuracy using this was between 77 - 80%. As part of our analysis, we see that one model is generally better than the other, so we will use that one. 
# 
# If we further analyze our dataset, one thing we noticed when we are buidling our model was that there are certain words that appear more than others. This can be visualized with our line plot; the spikes in the graph represent words that are used more than others. Taking a closer analysis at these words, we notice that many of them are neutral words, and thus, only creates noises in our model. By taking those values out, we rerun our model and notice that are R^2 values are higher. Thus, this means that by removing unnecessary noise, we are improving our model. 
# 
# **Presentation/Visualizations** Throughout the process, we used graphs to convey the relationship between our paramters and c, as well as visualize frequency. The final product that we've created is a function in which we can input a text or comment, and our function will use the model that we created to predict the connotation of our comment with an almost 80% accuracy. It will use the past train posts to determine whether our commment is positive or negative based off of the frequency of speicifc words. As can be seen in the example, when we input a positive comment, the function returns "positive" and when we put in a negative comment, it returns "negative."

# ## Challenge Problem: Automated Medical Diagnosis
# 
# In this problem, you are going to build a model to diagnose heart disease. 
# 
# The training set is provided in the file ``dataset_3_train.txt`` and there are two test sets: ``dataset_3_test_1.txt`` and ``dataset_3_test_2.txt``. Each patient in the datasets is described by 5 biomarkers extracted from cardiac SPECT images; the last column in each dataset contains the disease diagnosis (1 indicates that the patient is normal, and 0 indicates that the patient suffers from heart disease).
# 
# - Fit a logistic regression model to the training set, and report its accuracy on both the test sets. 
# 
# 
# - Is your accuracy rate meaningful or reliable? How comfortable would you be in using your predictions to diagnose real living patients? Justify your answers. 
# 
#   (**Hint:** How does the performance of your model compare with a classifier that lumps all patients into the same class?)
# 
# 
# - Let's call the logistic regression model you learned, ${C}_1$. Your colleague suggests that you can get higher accuracies for this task by using a threshold of 0.05 on the Logistic regression model to predict labels instead of the usual threshold of 0.5, i.e. use a classifier that predicts 1 when $\widehat{P}(Y = 1\,|\, X) \geq 0.05$ and 0 otherwise. Let's call this classifier ${C}_2$. Does ${C}_2$ perform better the two test sets - that is, which one would you rather use for automated diagnostics? Support your conclusion with careful analysis. 
# 
# 
# - Generalize your analysis of these two classifiers. Under what general conditions does $C_2$ perform better than ${C}_1$? Support your conclusion with a mathematical proof or simulation
# 
# 
# **Hint:** You were told in class that a classifier that predicts 1 when $\widehat{P}(Y = 1 \,|\, X) \geq 0.5$, and 0 otherwise, is the Bayes classifier. This classifier minimizes the classification error rate. What can you say about a classifier that uses a threshold other than $0.5$? Is it the Bayes classifier for a different loss function?
# 
# 
# **Hint:** For the first three parts, you might find it useful to analyze the conditional accuracy on each class.

# **Note: In this problem I swapped 0 to mean healthy and 1 to mean the patient suffers from heart disease. **

# In[61]:

#Import data sets 
dt_3_train = np.loadtxt("dataset_3_train.txt" , delimiter = ',')
dt_3_test1 = np.loadtxt("dataset_3_test_1.txt", delimiter = ',')
dt_3_test2 = np.loadtxt("dataset_3_test_2.txt", delimiter = ',')


# In[182]:

#Split data into train and test
x_train = dt_3_train[:,0:4]
y_train = dt_3_train[:,-1]
x_test1 = dt_3_test1[:,0:4]
y_test1 = dt_3_test1[:,-1]
x_test2 = dt_3_test2[:,0:4]
y_test2 = dt_3_test2[:,-1]


# In[183]:

#Test for the best regualrizing paramter
l1_acc = []
l2_acc = []
for i in xrange(-7, 8):
    c = 10**i
    l1 = cross_val_l1(c, x_train, y_train)
    l1_acc.append(l1)
    l2 = cross_val_l2(c, x_train, y_train)
    l2_acc.append(l2)
    

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.semilogx(10.0**np.arange(-7,8), 
            l1_acc,
            label = 'Lasso')
ax.semilogx(10.0**np.arange(-7, 8), 
            l2_acc, label = 'Ridge')

ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Paramters')
ax.legend(loc='best')
ax.set_ylim (.9, 1)

plt.show()


# In[184]:

#Calculate the best paramter
max_indexl1 = np.argmax(l1_acc) - 7
max_accl1 = np.max(l1_acc)
max_indexl2 = np.argmax(l2_acc) - 7
max_accl2 = np.max(l2_acc)
print 'According to Lasso, the best paramter is 10^' +str(max_indexl1) + 'and the max accuracy is ' + str(max_accl1)
print 'According to Ridge, the best paramter is 10^' + str(max_indexl2) + 'and the max accuracy is ' + str(max_accl2)


# In[187]:

#Test accuracy on each of the test sets
logit_l1 = LogReg(C = 10**max_indexl1, penalty = "l1")
logit_11 = logit_l1.fit(x_train, y_train)
acc_l1_test1 = logit_11.score(x_test1, y_test1)
acc_l1_test2 = logit_l1.score(x_test2, y_test2)

logit_l2 = LogReg(C = 10**max_indexl2, penalty = "l2")
logit_12 = logit_l2.fit(x_train, y_train)
acc_l2_test1 = logit_l2.score(x_test1, y_test1)
acc_l2_test2 = logit_l2.score(x_test2, y_test2)
print 'The accuracy for test 1 using Lasso is ' + str(acc_l1_test1) + " and the accuracy using Ridge is " + str(acc_l2_test1)
print 'The accuracy for test 2 using Lasso is ' + str(acc_l1_test2) + " and the accuracy using Ridge is " + str(acc_l2_test2)


# **- Is your accuracy rate meaningful or reliable? How comfortable would you be in using your predictions to diagnose real living patients? Justify your answers. **
# 
# It's not reliable and very unsettling. Especially on test set two, it's terrifying if our model predicts something as dire as health related issues as accurately as a flip of a coin. Also, it's not very meaningful, because if we have a false negative and say that someone is healthy when they are not, then this much worse than a false positive. Thus, we should not be treating every wrong answer the same; rather we should be more concerned with the false negatives. In a greater scope, this means that accuracy of our model is not necessarily the indicator of a good model. 

# In[66]:

#We notice that when we regularize we drive all our probabilities to .5.
logit_l1 = LogReg(C = 10**max_indexl1, penalty = "l1")
logit_11 = logit_l1.fit(x_train, y_train)
c_probabilites = logit_11.predict_proba(x_test1)


# In[83]:

###Because this is not what we want, we will use regular logistic regularization
###Then, we will predict patients to be healthy only if their probability of being unhealthy is less than .05
#Run a logistic regression and get the probabilities of each x_test value
logit_l1 = LogReg()
logit_11 = logit_l1.fit(x_train, y_train)
predict_probabilities1 = logit_11.predict_proba(x_test1)
predict_probabilities2 = logit_l1.predict_proba(x_test2)
pred_ytest1 = []
test1_acc = 0
pred_ytest2 = []
test2_acc = 0

#Loop through each test sets probabilites and assign the proper values. Add the correct values to get accuracy
for i in xrange(0, (x_test1.shape[0])):
    if(predict_probabilities1[i][1] <= .05):
        pred_ytest1.append(1)
    else:
        pred_ytest1.append(0)
    if (pred_ytest1[i] == y_test1[i]):
        test1_acc +=1

for j in xrange(0, (x_test2.shape[0])):
    if(predict_probabilities2[j][1] <= .05):
        pred_ytest2.append(1)
    else:
        pred_ytest2.append(0)
    if (pred_ytest2[j] == y_test[j]):
        test2_acc += 1
    
print 'With the new accuracy conditions, our test 1 accuracy score is ' + str(test1_acc*1.0/y_test1.shape[0])
print 'With the new accuracy conditions, our test 2 accuracy score is ' + str(test2_acc*1.0/y_test2.shape[0])



# **Discussion** In both cases, our new classifiers performed worse relative to our original classifiers. However, our new classifier will perform better when there is a significant number of false negatives. This is, there is a greater number of unhealthy participants relative to healthy participants. This is because our model is more robust against saying people are healthy when they are unhealthy; if there are more unhealthy people, then we would perform better. Therefore, I would prefer to use C_2 because it will prevent us from diagnosing people who are unhealthy as healthy, which is more important. 
# 
# In a general sense, the C_1 classifier is more accurate in terms of getting the most cases correct if the cases are randomly aassigned and there are roughly equal number of healthy and unhealthy patients. That is, the classifier is equal to .5. In the C_2 classifier, we would be more accurate if the classifier of P(healthy) was closer to our threshold of .05. 

# In[ ]:



