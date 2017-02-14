
# coding: utf-8

# # CS 109A/AC 209A/STAT 121A Data Science: Homework 6
# **Harvard University**<br>
# **Fall 2016**<br>
# **Instructors: W. Pan, P. Protopapas, K. Rader**<br>
# **Due Date: ** Wednesday, November 2nd, 2016 at 11:59pm

# Download the `IPython` notebook as well as the data file from Vocareum and complete locally.
# 
# To submit your assignment, in Vocareum, upload (using the 'Upload' button on your Jupyter Dashboard) your solution to Vocareum as a single notebook with following file name format:
# 
# `last_first_CourseNumber_HW6.ipynb`
# 
# where `CourseNumber` is the course in which you're enrolled (CS 109a, Stats 121a, AC 209a). Submit your assignment in Vocareum using the 'Submit' button.
# 
# **Avoid editing your file in Vocareum after uploading. If you need to make a change in a solution. Delete your old solution file from Vocareum and upload a new solution. Click submit only ONCE after verifying that you have uploaded the correct file. The assignment will CLOSE after you click the submit button.**
# 
# Problems on homework assignments are equally weighted. The Challenge Question is required for AC 209A students and optional for all others. Student who complete the Challenge Problem as optional extra credit will receive +0.5% towards your final grade for each correct solution. 

# Import libraries

# In[483]:

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import mode
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression as LogReg
import matplotlib
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.neighbors import KNeighborsRegressor as KNN
get_ipython().magic(u'matplotlib inline')


# ## Problem 0: Basic Information
# 
# Fill in your basic information. 
# 
# ### Part (a): Your name

# Hsu, Curtis

# ### Part (b): Course Number

# CS109a

# ### Part (c): Who did you work with?

# [First and Land names of students with whom you have collaborated]

# **All data sets can be found in the ``datasets`` folder and are in comma separated value (CSV) format**

# ## Problem 1: Recommender System for Movies
# 
# In this problem, you will build a model to recommend movies using ratings from users. 
# 
# The dataset for this problem is contained in `dataset_4_ratings.txt`. This dataset contains ratings from 100 users for 1000 movies. The first two columns contain the user and movie IDs. The last column contains a 1 if the user liked the movie, and 0 otherwise. Not every movie is rated by every user (i.e. some movies have more ratings than others).
# 
# The names of the movies corresponding to the IDs are provided in `dataset_4_movie_names.txt`.
# 
# ### Part 1(a): Exploring how to rank
# 
# One way of recommending movies is to recommend movies that are generally agreed upon to be good. But how do we measure the "goodness" or "likability" of a movie?
# 
# 
# - **Implementation:** Suppose we measure the "goodness" of a movie by the probability that it will be liked by a user, $P(\textbf{label} = \text{like}|\textbf{movie}) = \theta_{\text{movie}}$. Assuming that each user independently rates a given movie according to the probability $\theta_{\text{movies}}$. Use a reasonable estimate of $\theta_{\text{movies}}$ to build a list of top 25 movies that you would recommend to a new user.
# 
#    **Hint:** What does the likelihood function, $P(\textbf{likes} = k | \theta_{\text{movie}}, n, \textbf{movie})$, look like? What $\theta_{\text{movie}}$ will maximize the likelihood?
#    
# 
# - **Analysis:** Why is using $\theta_{\text{movie}}$ to rank movies more appropriate than using the total number of likes? Explain why your estimate of $\theta_{\text{movie}}$ is reasonable. Explain the potential draw backs of estimating $\theta_{\text{movie}}$ this way.
# 
#    **Hint:** Under what conditions may models that maximize the likelihood be suboptimal? Do those conditions apply here?   

# In[484]:

ratings_df = pd.read_csv('datasets/dataset_4_ratings.txt', delimiter=',')
ratings_df.head()


# In[485]:

names_df = pd.read_csv('datasets/dataset_4_movie_names.txt', delimiter='|')
names_df.head()


# In[486]:

#--------  movie_stats
# A function that extracts the number of likes and total number of ratings for a movie
# Input: 
#      movie_name (an optional parameter containing the exact name of the movie)
#      movie_name_contains (an optional parameter containing a portion of the name of the movie)
# Returns: 
#      total_ratings (the total number of ratings for a movie)
#      likes (the total number of likes for a movie)

def movie_stats(movie_name=None, movie_name_contains=None):
    
    #If given an exact movie name:
    if movie_name is not None:        
        #Find the index of the movie, by name, in the "names" dataframe
        movie_index = names_df[names_df['movie_name'] == movie_name].index[0]
        #Get the id for the movie in the "names" dataframe
        movie_id = names_df.loc[movie_index, 'movie_id']
        #Get all ratings for the movie, by id, in the "ratings" dataframe
        ratings_for_movie = ratings_df[ratings_df['movie_id'] == movie_id]
        #Count the total number of ratings
        total_ratings = len(ratings_for_movie)
        #Count the likes (the 1's)
        likes = ratings_for_movie['rating'].sum()
    
    #Otherwise, if given a partial movie name:
    elif movie_name_contains is not None:
        #Find the index of the movie, by name, in the "names" dataframe
        movie_index = names_df[names_df['movie_name'].str.contains(movie_name_contains)].index[0]
        #Get the id for the movie in the "names" dataframe
        movie_id = names_df.loc[movie_index, 'movie_id']
        #Get all ratings for the movie, by id, in the "ratings" dataframe
        ratings_for_movie = ratings_df[ratings_df['movie_id'] == movie_id]
        #Count the total number of ratings
        total_ratings = len(ratings_for_movie)
        #Count the likes (the 1's)
        likes = ratings_for_movie['rating'].sum()
    
    else:
        total_ratings = 0.
        likes = 0.
    
    return float(total_ratings), likes


# In[487]:

total_ratings, likes = movie_stats(movie_name_contains='Toy Story')

print 'total number of ratings for Toy Story:', total_ratings
print 'number of likes for Toy Story:', likes


# In[488]:

total_ratings, likes = movie_stats(movie_name_contains="Shawshank Redemption")

print 'total number of ratings for Star Wars:', total_ratings
print 'number of likes for Star Wars:', likes


# In[489]:

total_ratings, likes = movie_stats(movie_name_contains='French Twist')

print 'total number of ratings for French Twist:', total_ratings
print 'number of likes for French Twist:', likes


# In[490]:

#Make a list of movie names and their ratings info
likability = []

#Iterate through all the movie names
for name in names_df['movie_name'].values:
    #Get ratings info for movie
    total_ratings, likes = movie_stats(movie_name=name)
    #Add movie info to our list
    likability.append((name, likes, total_ratings, likes / total_ratings))

#Sort our list of movie info by like-percentage, in descending order
sorted_likability = sorted(likability, key=lambda t: t[3], reverse=True)  
#Get the movies with top 25 like-percentage
top_25_movies = sorted_likability[:25]

#Print results of ranking
print 'Top 25 Movies'
print '****************************'
for movie, likes, total_ratings, likable in top_25_movies:
    print movie, ':', likable, '({}/{})'.format(likes, total_ratings)


# **PART A: Question** What does the likelihood function, $P(\textbf{likes} = k | \theta_{\text{movie}}, n, \textbf{movie})$, look like? What $\theta_{\text{movie}}$ will maximize the likelihood? Why is using $\theta_{\text{movie}}$ to rank movies more appropriate than using the total number of likes? Explain why your estimate of $\theta_{\text{movie}}$ is reasonable. Explain the potential draw backs of estimating $\theta_{\text{movie}}$ this way.
# 
# **Solution:** The likelihood function looks like the probability that we observe k amount of likes given a movie out of y amount of ratings. The theta that will maximize the likelihood is thus the naive definition of probability. To maximize the probability that we will observe x likes out of y ratings, then our MLE for theta would be x/y. 
# 
# Using theta to rank movies is more appropriate than using the total number of likes because more popular movies will have more opportunities to gain likes, and thus may have more likes than a less popular yet higher quality movie. Our estimate of theta is reasonable because we are able to use past observations to predict the likelihood that a new person will like a movie. The reason we do this is because, when using enough observations, we are assuming that movies that receive more likes are inherently better. If everyone that has previously watched a movie liked it, then it is a fair estimate that people watching after will likely also like the movie. Thus it would make sense that the probability that someone likes a movie is equal to the history of that movie's ratings. 
# 
# A potential drawback of estmating theta this way is that movies that are not that popular and have not received that many ratings will have artifically high or low thetas, and thus not accurately represent how good the movie is. For example, if a low quality movie was only watched once but it was watched by someone who particularly liked the movie, then its rating will be 1, and thus our estimate for theta will be one, but this may not be accurate 

# ### Part 1(b): Exploring the effect of prior beliefs
# 
# Let's add a prior, $p(\theta_{\text{movie}})$, to our probabilistic model for movie rating. To keep things simple, we will restrict ourselves to using beta priors.
# 
# - **Analysis:** How might adding a prior to our model benifit us in our specific task? Why are beta distributions appropriate priors for our application?
# 
#   **Hint:** Try visualizing beta priors $a = b = 1$, $a = b = 0.5$, $a = b = 2$ and $a = 4, b = 2$, for example, what kind of plain-English prior beliefs about the movie does each beta pdf encode?
# 
# 
# - **Implementation/Analysis:** How does the choice of prior affect the posterior distribution of the 'likability' for the movies: *Toy Story, Star Wars, The Shawshank Redemption, Down Periscope and Chain Reaction*.
# 
#    **Hint:** Use our posterior sampling function to visualize the posterior distribution.
#    
#  
# - **Implementation/Analysis:** How does the effect of the prior on the posterior distribution vary with the number of user ratings? 
# 
#    **Hint:** Visualize the posterior distribution for different sizes of subsample of user ratings for the movie *Star Wars*.
#    
# In the following, we've provide you a couple of functions for visualize beta priors and approximating their associated posteriors.

# In[491]:

#--------  plot_beta_prior
# A function to visualize a beta pdf on a set of axes
# Input: 
#      a (parameter controlling shape of beta prior)
#      b (parameter controlling shape of beta prior)
#      color (color of beta pdf)
#      ax (axes on which to plot pdf)
# Returns: 
#      ax (axes with plot of beta pdf)

def plot_beta_prior(a, b, color, ax):
    rv = sp.stats.beta(a, b)
    x = np.linspace(0, 1, 100)
    ax.plot(x, rv.pdf(x), '-', lw=2, color=color, label='a=' + str(a) + ', b=' + str(b))
    ax.set_title('Beta prior with a=' + str(a) + ', b=' + str(b))
    ax.legend(loc='best')
    return ax


# In[492]:

#--------  sample_posterior
# A function that samples points from the posterior over a movie's 
# likability, given a binomial likelihood function and beta prior
# Input: 
#      a (parameter controlling shape of beta prior)
#      b (parameter controlling shape of beta prior)
#      likes (the number of likes in likelihood)
#      ratings (total number of ratings in likelihood)
#      n_samples (number of samples to take from posterior)
# Returns: 
#      post_samples (a array of points from the posterior)

def sample_posterior(a, b, likes, ratings, n_samples):
    post_samples = np.random.beta(a + likes, b + ratings - likes, n_samples)
    return post_samples


# In[493]:

#--------  find_mode
# A function that approximates the mode of a distribution given a sample from the distribution
# Input: 
#      values (samples from the distribution)
#      num_bins (number of bins to use in approximating histogram)
# Returns: 
#      mode (the approximate mode of the distribution)

def find_mode(values, num_bins):
    
    #Make an approximation (histogram) of the distribution using the samples
    bins, edges = np.histogram(values, bins=num_bins)
    #Find the bin in the histogram with the max height
    max_height_index = np.argmax(bins)
    #Find the sample corresponding to the bin with the max height (the mode)
    mode = (edges[max_height_index] + edges[max_height_index + 1]) / 2.
    
    return mode


# In[494]:

#A list of beta distribution shapes to try out
beta_shapes = [(1, 1), (0.5, 0.5), (2, 2), (4, 2), (2, 4)]
#Length of the list of shapes
n = len(beta_shapes)

#Plot all the beta pdfs in a row
fig, ax = plt.subplots(1, n, figsize=(20, 4))

#Start the index of the current subplot at 0
ax_ind = 0
#Iterate through all the shapes
for a, b in beta_shapes:
    #Plot the beta pdf for a particular shape
    plot_beta_prior(a, b, 'blue', ax[ax_ind])
    #Increment the subplot index
    ax_ind += 1
    
plt.tight_layout()    
plt.show() 


# In[495]:

#Get the name of the first movie in the top 25 list
movie_name = top_25_movies[0][0]

#Get the ratings info for the first movie in the top 25 list
likes = top_25_movies[0][1]
total_ratings = top_25_movies[0][2]
likability = top_25_movies[0][3]

#Print movie info
print '{}: {} ({}/{})'.format(movie_name, likability, likes, total_ratings)

#Number of samples to use when approximating our posterior
n_samples = 10000

#Plot the posterior corresponding to each prior
fig, ax = plt.subplots(1, n, figsize=(20, 4))

#Start the index of the current subplot at 0
ax_ind = 0

#Iterate through all the shapes
for a, b in beta_shapes:   
    #Draw samples from the posterior corresponding to a particular beta prior
    post_samples = sample_posterior(a, b, likes, total_ratings, n_samples)
    #Approximate the posterior with a histogram of these samples
    ax[ax_ind].hist(post_samples, bins=30, color='red', alpha=0.5)
    #Find the approximate mode of the posterior
    mode = find_mode(post_samples, 30)
    #Plot the mode as a vertical line
    ax[ax_ind].axvline(x=mode, linewidth=3, label='Posterior mode')
    
    #Set title, legends etc
    ax[ax_ind].set_title('Posterior, with Beta prior (a={}, b={})'.format(a, b))
    ax[ax_ind].legend(loc='best')
    #Increment the subplot index
    ax_ind += 1

plt.tight_layout()
plt.show() 


# In[496]:

movie_name_contains = 'Toy Story'
movie_index = names_df[names_df['movie_name'].str.contains(movie_name_contains)].index[0]
movie_id = names_df.loc[movie_index, 'movie_id']
#Find the index of the movie, by name, in the "names" dataframe
movie_index = names_df[names_df['movie_name'].str.contains(movie_name_contains)].index[0]
#Get the id for the movie in the "names" dataframe
movie_id = names_df.loc[movie_index, 'movie_id']
#Get all ratings for the movie, by id, in the "ratings" dataframe
ratings_for_movie = ratings_df[ratings_df['movie_id'] == movie_id]
#Count the total number of ratings
total_ratings = len(ratings_for_movie)
#Count the likes (the 1's)
likes = ratings_for_movie['rating'].sum()


# In[497]:

#Calculates the effects of betas on a movie 
#input is title, output is visualizations of effect of betas
def beta_effect(movie_name_contains=None):

    movie_index = names_df[names_df['movie_name'].str.contains(movie_name_contains)].index[0]
    movie_name = names_df.loc[movie_index, 'movie_name']
    #Get the id for the movie in the "names" dataframe
    movie_id = names_df.loc[movie_index, 'movie_id']
    #Get all ratings for the movie, by id, in the "ratings" dataframe
    ratings_for_movie = ratings_df[ratings_df['movie_id'] == movie_id]
    #Count the total number of ratings
    total_ratings = len(ratings_for_movie)
    #Count the likes (the 1's)
    likes = ratings_for_movie['rating'].sum()
    likability = likes*1.0/total_ratings

    #Print movie info
    print '{}: {} ({}/{})'.format(movie_name, likability, likes, total_ratings)

    #Number of samples to use when approximating our posterior
    n_samples = 10000

    #Plot the posterior corresponding to each prior
    fig, ax = plt.subplots(1, n, figsize=(20, 4))

    #Start the index of the current subplot at 0
    ax_ind = 0

    #Iterate through all the shapes
    for a, b in beta_shapes:   
        #Draw samples from the posterior corresponding to a particular beta prior
        post_samples = sample_posterior(a, b, likes, total_ratings, n_samples)
        #Approximate the posterior with a histogram of these samples
        ax[ax_ind].hist(post_samples, bins=30, color='red', alpha=0.5)
        #Find the approximate mode of the posterior
        mode = find_mode(post_samples, 30)
        #Plot the mode as a vertical line
        ax[ax_ind].axvline(x=mode, linewidth=3, label='Posterior mode')

        #Set title, legends etc
        ax[ax_ind].set_title('Posterior, with Beta prior (a={}, b={})'.format(a, b))
        ax[ax_ind].legend(loc='best')
        #Increment the subplot index
        ax_ind += 1

    plt.tight_layout()
    plt.show() 


# In[498]:

#Loop through movies given and visualize the outputs
movie_beta_lst = ['Toy Story', 'Star Wars', 'Shawshank Redemption', 'Down Periscope', 'Chain Reaction'] 
for i in range(0, len(movie_beta_lst)):
    beta_effect(movie_beta_lst[i])


# In[499]:

#Find the effect of betas on different subset size
def beta_effect_sub(size, movie_name_contains=None):

    movie_index = names_df[names_df['movie_name'].str.contains(movie_name_contains)].index[0]
    movie_name = names_df.loc[movie_index, 'movie_name']
    #Get the id for the movie in the "names" dataframe
    movie_id = names_df.loc[movie_index, 'movie_id']
    #Get all ratings for the movie, by id, in the "ratings" dataframe
    ratings_for_movie = ratings_df[ratings_df['movie_id'] == movie_id]
    subratings_for_movie = ratings_for_movie.iloc[0:size,:]
    #Count the total number of ratings
    total_ratings = len(subratings_for_movie)
    #Count the likes (the 1's)
    likes = subratings_for_movie['rating'].sum()
    likability = likes*1.0/total_ratings

    #Print movie info
    print '{}: {} ({}/{})'.format(movie_name, likability, likes, total_ratings)
    print 'Subsize = ' + str(size)

    #Number of samples to use when approximating our posterior
    n_samples = 10000

    #Plot the posterior corresponding to each prior
    fig, ax = plt.subplots(1, n, figsize=(28, 6))

    #Start the index of the current subplot at 0
    ax_ind = 0

    #Iterate through all the shapes
    for a, b in beta_shapes:   
        #Draw samples from the posterior corresponding to a particular beta prior
        post_samples = sample_posterior(a, b, likes, total_ratings, n_samples)
        #Approximate the posterior with a histogram of these samples
        ax[ax_ind].hist(post_samples, bins=30, color='red', alpha=0.5)
        #Find the approximate mode of the posterior
        mode = find_mode(post_samples, 30)
        #Plot the mode as a vertical line
        ax[ax_ind].axvline(x=mode, linewidth=3, label='Posterior mode')

        #Set title, legends etc
        ax[ax_ind].set_title('Posterior, with Beta prior (a={}, b={})'.format(a, b), fontsize = 14)
        ax[ax_ind].legend(loc='best')
        #Increment the subplot index
        ax_ind += 1

    plt.tight_layout()
    plt.show() 


# In[500]:

for i in range(7,67, 10):
    beta_effect_sub(i, 'Star Wars')


# **Part 1(b):Questions and Solutions**
# 
# Let's add a prior, $p(\theta_{\text{movie}})$, to our probabilistic model for movie rating. To keep things simple, we will restrict ourselves to using beta priors.
# 
# - **Analysis:** How might adding a prior to our model benifit us in our specific task? Why are beta distributions appropriate priors for our application?
# 
#   **Hint:** Try visualizing beta priors $a = b = 1$, $a = b = 0.5$, $a = b = 2$ and $a = 4, b = 2$, for example, what kind of plain-English prior beliefs about the movie does each beta pdf encode?
# 
# 
# - **Implementation/Analysis:** How does the choice of prior affect the posterior distribution of the 'likability' for the movies: *Toy Story, Star Wars, The Shawshank Redemption, Down Periscope and Chain Reaction*.
# 
#    **Hint:** Use our posterior sampling function to visualize the posterior distribution.
#    
#  
# - **Implementation/Analysis:** How does the effect of the prior on the posterior distribution vary with the number of user ratings? 
# 
#    **Hint:** Visualize the posterior distribution for different sizes of subsample of user ratings for the movie *Star Wars*.
#    
# **Analysis**: Adding a prior to our model benefits us because it allows up to draw our estimate of our thetas closer to the true probability that someone will like a movie. The beta distributions are appropriate priors for our application because a movie is either liked or not like, so it would make sense to draw the estimate closer to .5. This makes sense because, it does not make much sense to say that someone has 100% chance of liking a movie because prior 1 out of 1 people did. Thus, regualization helps us get the accurate population value. 
# 
# **Implementation/Analysis**: The choice of prior affects the posterior distribution of the likability by drawing them to specific probabilities. For example: a = b = 1 means that our estimate is the true proportion. Our a = b = .5 means that the true proportion is on the edges. This means that, for Star Wars, the true proporition is drawn towards 1 and for Down Periscope it would be drawn towards 0. a = b = 2 means that the scores should be normally distributed and draws them to the middle, For a = 4 and b = 2, the scores are drawn towards .8 and for a = 2 b = 4 they are drawn towards .2. The impact of each beta combination can be seen in the visualizations above.
# 
# **Implementation/Analysis**: The effect of the prior on the posterior varies in the sense that the greater the number of user ratings, the less the impact of the prior on the posterior. This can be seen in the visualizations with starwars. As the subsize increases, the impacts of the paramters a and b caused the mode to deviate less from the observed estimate of theta. 

# ### Part 1(c): Recommendation based on ranking
# 
# - **Implementation:** Choose a reasonable beta prior, choose a reasonable statistic to compute from the posterior, and then build a list of top 25 movies that you would recommend to a new user based on your chosen posterior statistic.  
# 
#  
# - **Analysis:** How does your top 25 movies compare with the list you obtained in part(a)? Which method of ranking is better?
# 
#  
# - **Analysis:** So far, our estimates of the 'likability' for a movie was based on the ratings provided by all users. What can be the draw back of this method? How can we improve the recommender system for individual users (if you feel up to the challenge, implement your improved system and compare it to the one you built in the above)? 

# In[501]:

#Choose a beta prior that encodes a reasonable belief about likability 
a = 2 
b = 2

#Make a list of movie names and their ratings info
likability = []

#Iterate through all the movie names
for name in names_df['movie_name'].values:
    #Get ratings info for movie
    total_ratings, likes = movie_stats(movie_name=name)
    #Approximate the posterior given the ratings info and the prior
    post_samples = sample_posterior(a, b, likes, total_ratings, n_samples)
    #Approximate posterior mode
    mode = find_mode(post_samples, 30)
    #Add movie info to our list
    likability.append((name, likes, total_ratings, mode))

#Sort our list of movie info by like-percentage, in descending order
sorted_likability = sorted(likability, key=lambda t: t[3], reverse=True)  
#Get the movies with top 25 like-percentage
top_25_movies = sorted_likability[:25]

#Print results of ranking
print 'Top 25 Movies'
print '****************************'
for movie, likes, total_ratings, likable in top_25_movies:
    print movie, ':', likable, '({}/{})'.format(likes, total_ratings)


# ** Part 1(c): Recommendation based on ranking - Questions and Answers**
# - **Analysis:** How does your top 25 movies compare with the list you obtained in part(a)? Which method of ranking is better?
# 
#  
# - **Analysis:** So far, our estimates of the 'likability' for a movie was based on the ratings provided by all users. What can be the draw back of this method? How can we improve the recommender system for individual users (if you feel up to the challenge, implement your improved system and compare it to the one you built in the above)? 
# 
# 
# **Analysis** Our top 25 movies are different relative to part(a) in the sense that the movies are different because the estimated thetas are now different. I think that the ranking method of part c is better because simply using the proportion of likes gives movies that aren't as popular an advantage because they are more likely to 1/1 people liking the movie. Thus, by drawing the estimated thetas closer to a normal distribution, and allowing movies with more ratings to be less affected by this, produces a better ranking system. 
# 
# **Analysis** One drawback of our method is that each user has the same 'vote.' That is, whether they are an advent movie watcher or not does not influence their say on a movie. This is a drawback because movie aficionados and amateurs should not have the same weight. To improve the recommender system, we could give greater weight to user_ids that give more ratings, because we can assume that people who watch more movies can give more accurate responses. Another drawback that would be more difficult to implement is that we should be, for users who seem to gravitate towards a specific genre, we could alter their priors priors. For example, if a user_id has shown to like action movies, then we should priors that reflect this. 

# ---

# ## Problem 2: Predicting Urban Demographic Changes
# 
# ### Part 2(a): Temporal patterns in urban demographics
# 
# In this problem you'll work with some neighborhood demographics of a region in Boston from the years 2000 to 2010. 
# 
# The data you need are in the files `dataset_1_year_2000.txt`, ..., `dataset_1_year_2010.txt`. The first two columns of each dataset contain the adjusted latitude and longitude of some randomly sampled houses. The last column contains economic status of a household: 
# 
# 0: low-income, 
# 
# 1: middle-class, 
# 
# 2: high-income 
# 
# Due to the migration of people in and out of the city, the distribution of each economic group over this region changes over the years. The city of Boston estimates that in this region there is approximately a 25% yearly increase in high-income households; and a 25% decrease in the remaining population, with the decrease being roughly the same amongst both the middle class and lower income households.
# 
# Your task is to build a model for the city of Boston that is capable of predicting the economic status of a household based on its geographical location. Furthermore, your method of prediction must be accurate over time (through 2010 and beyond). 
# 
# **Hint:** look at data only from 2000, and consider using both Linear Discriminant Analysis (LDA) and Logistic Regression. Is there a reason one method would more suited than the other for this task?
# 
# **Hint:** how well do your two models do over the years? Is it possible to make use of the estimated yearly changes in proportions of the three demographic groups to improve the predictive accuracy of each models over the years? 
# 
# To help you visually interpret and assess the quality of your classifiers, we are providing you a function to visualize a set of data along with the decision boundaries of a classifier.

# In[502]:

#Input the year of the dataset 
#Loads the data and breaks into train and test data and x and y
#Outputs x_train, y_train, x_test, and y_test
def load_data(num):
    dt = 'datasets/dataset_1_year_' + str(num) + ".txt"
    dt_year = np.loadtxt(dt)
    x = dt_year[:,0:2]
    y = dt_year[:,-1]
    train, test = sk_split(dt_year, train_size = .7)
    x_train = train[:,0:2]
    y_train = train[:,-1]
    x_test = test[:,0:2]
    y_test = test[:,-1]
    return x_train, y_train, x_test, y_test, x, y
    


# In[503]:

#Load all the data
xtrain_2000, ytrain_2000, xtest_2000, ytest_2000, x2000, y2000 = load_data(2000)
xtrain_2001, ytrain_2001, xtest_2001, ytest_2001, x2001, y2001 = load_data(2001)
xtrain_2002, ytrain_2002, xtest_2002, ytest_2002, x2002, y2002 = load_data(2002)
xtrain_2003, ytrain_2003, xtest_2003, ytest_2003, x2003, y2003 = load_data(2003)
xtrain_2004, ytrain_2004, xtest_2004, ytest_2004, x2004, y2004 = load_data(2004)
xtrain_2005, ytrain_2005, xtest_2005, ytest_2005, x2005, y2005 = load_data(2005)
xtrain_2006, ytrain_2006, xtest_2006, ytest_2006, x2006, y2006 = load_data(2006)
xtrain_2007, ytrain_2007, xtest_2007, ytest_2007, x2007, y2007 = load_data(2007)
xtrain_2008, ytrain_2008, xtest_2008, ytest_2008, x2008, y2008 = load_data(2008)
xtrain_2009, ytrain_2009, xtest_2009, ytest_2009, x2009, y2009 = load_data(2009)
xtrain_2010, ytrain_2010, xtest_2010, ytest_2010, x2010, y2010 = load_data(2010)


# In[504]:

#Function that predicts the priors based on year
#Input: A year after 2000
#Outputs the priors
prior_2000_2 = len(y2000[y2000 == 2])
prior_2000_1 = len(y2000[y2000 == 1])
prior_2000_0 = len(y2000[y2000 == 0])

def gen_prior(year):
    prior_lst = []
    multiples = int(year) - int(2000)
    high_income = prior_2000_2
    mid_income = prior_2000_1
    low_income = prior_2000_0
    
    #Calculate the new proportions of a given year
    new_high = int(high_income*(1.25**multiples))
    mid_decrease = int(.5*(new_high - high_income))
    low_decrease = mid_decrease
    new_mid = mid_income - mid_decrease
    new_low = low_income - low_decrease
    
    high_income_perc = (new_high * 1.0) /(new_high + new_mid + new_low)
    mid_income_perc = (new_mid * 1.0) / (new_high + new_mid + new_low)
    low_income_perc = (new_low * 1.0) / (new_high + new_mid + new_low)
    prior_lst.append(mid_income_perc)
    prior_lst.append(low_income_perc)
    prior_lst.append(high_income_perc)
    return prior_lst
        


# In[505]:

#Let's explore how each data set looks from 2000 to 2010
for i in range(0, 11):
    fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    year = str(2000 + i)
    xtrain, ytrain, xtest, ytest, x, y = load_data(year)
    logit = LogReg()
    logit = logit.fit(xtrain, ytrain)
    logit_score = logit.score(xtest, ytest)
    plot_decision_boundary(x, y, logit, False, 'Logit - Year: ' + str(year) + ' Accuracy = ' + str(logit_score), ax1)
    lda = da.LinearDiscriminantAnalysis()
    lda = lda.fit(xtrain, ytrain)
    lda_score = lda.score(xtest, ytest)
    plot_decision_boundary(x, y, lda, False, 'LDA - Year: ' + str(year) + ' Accuracy = ' + str(lda_score), ax2)
    qda = da.QuadraticDiscriminantAnalysis()
    qda.fit(xtrain, ytrain)
    qda_score = qda.score(xtest, ytest)
    plot_decision_boundary(x, y, qda, False, 'QDA - Year: ' + str(year) + 'Accuracy = ' + str(qda_score), ax3)
    


# When we try to predict future data, however, we will not have the luxury of getting the data of the city before running our regression. Thus, we must use our data from our most previous year to predict the future. To find which model is best, we will plot a logistic, LDA, and QDA model. For our LDA model, we can use the priors that we solved for before 

# In[506]:

#Now we will attempt to model accuracy of our years using fits from 2000 and our priors
for i in range(0, 11):
    fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    year = str(2000 + i)
    prior = gen_prior(year)
    xtrain, ytrain, xtest, ytest, x, y = load_data(2000)
    xtrain_year, ytrain_year, xtest_year, ytest_year, x_year, y_year = load_data(year)
    logit = LogReg()
    logit = logit.fit(x, y)
    logit_score = logit.score(x_year, y_year)
    plot_decision_boundary(x_year, y_year, logit, False, 'Logit - Year: ' + str(year) + ' Accuracy = ' + str(logit_score), ax1)
    lda = da.LinearDiscriminantAnalysis(priors = prior)
    lda = lda.fit(x, y)
    lda_score = lda.score(x_year, y_year)
    plot_decision_boundary(x_year, y_year, lda, False, 'LDA - Year: ' + str(year) + ' Accuracy = ' + str(lda_score), ax2)
    qda = da.QuadraticDiscriminantAnalysis(priors = prior)
    qda.fit(x, y)
    qda_score = qda.score(x_year, y_year)
    plot_decision_boundary(x_year, y_year, qda, False, 'QDA - Year: ' + str(year) + ' Accuracy = ' + str(qda_score), ax3)
    


# In[507]:

#Function that predicts the priors based on year
#Input: A year after 2010
#Outputs the priors

def gen_prior2010(year):
    #Create the baseline values
    prior_lst = []
    multiples = int(year) - int(2010)
    high_income = len(y2010[y2010 ==2])
    mid_income = len(y2010[y2010 ==1])
    low_income = len(y2010[y2010 ==0])
    
    #Calculate the changes
    new_high = int(high_income*(1.25**multiples))
    mid_decrease = int(.5*(new_high - high_income))
    low_decrease = mid_decrease
    new_mid = mid_income - mid_decrease
    new_low = low_income - low_decrease
    
    #Calculate the new priors
    high_income_perc = (new_high * 1.0) /(new_high + new_mid + new_low)
    mid_income_perc = (new_mid * 1.0) / (new_high + new_mid + new_low)
    low_income_perc = (new_low * 1.0) / (new_high + new_mid + new_low)
    prior_lst.append(mid_income_perc)
    prior_lst.append(low_income_perc)
    prior_lst.append(high_income_perc)
    return prior_lst


# **Solution**
# 
# From the information above, it is seen that LDA is the best model, as it surpases Logistic as years go on. It iperforms slightly better than QDA. Thus, we will use LDA to build our model that will predict populations after 2010. We will use our 2010 data to fit our model and our priors will be calculated using 2010 data. 

# In[508]:

#Input: The year of the incidence and the lattitude and longitude of the home
#Output: The predicted economic class of the family

def predict_econ(year, lat, lon):
    loc = np.array([lat, lon]).reshape(1,-1)
    xtrain, ytrain, xtest, ytest, x, y = load_data(2010)
    prior = gen_prior2010(year)
    lda = da.LinearDiscriminantAnalysis(priors = prior)
    lda.fit(x, y)
    ec_pred = lda.predict(loc)
    return ec_pred[0]


# In[509]:

##Test on 2008 data
predict_econ(2008, .1419822960552072078, .31490071164975)


# In[510]:

#--------  plot_decision_boundary
# A function that visualizes the data and the decision boundaries
# Input: 
#      x (predictors)
#      y (labels)
#      poly_flag (a boolean parameter, fits quadratic model if true, otherwise linear)
#      title (title for plot)
#      ax (a set of axes to plot on)
# Returns: 
#      ax (axes with data and decision boundaries)

def plot_decision_boundary(x, y, model, poly_flag, title, ax):
    # Plot data
    high = ax.scatter(x[y == 2, 0], x[y == 2, 1], c ='g')
    avg = ax.scatter(x[y == 1, 0], x[y == 1, 1], c='b')
    low = ax.scatter(x[y == 0, 0], x[y == 0, 1], c='r')
    
    # Create mesh
    interval = np.arange(0,1,0.01)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    # Predict on mesh points
    if(poly_flag):
        quad_features = preprocessing.PolynomialFeatures(degree=2)
        xx = quad_features.fit_transform(xx)
    yy = model.predict(xx)    
    yy = yy.reshape((n, n))

    # Plot decision surface
    x1 = x1.reshape(n, n)
    x2 = x2.reshape(n, n)
    ax.contourf(x1, x2, yy, alpha=0.1)
    
    # Label axes, set title
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.legend((high, avg, low), ("High", "Avg", "Low"), loc = 'best')
    
    return ax


# ### Part 2(b): Geographic patterns in urban demographics
# 
# In `dataset_2.txt` and `dataset_3.txt` you have the demographic information for a random sample of houses in two regions in Cambridge. There are only two economic brackets for the households in these datasets: 
# 
# 0: low-income or middle-class, 
# 
# 1 - high-income. 
# 
# For each region, recommend a classification model, chosen from all the ones you have learned, that is most appropriate for classifying the demographics of households in the region.
# 
# **Hint:** Support your answers with both numerical and visual analysis.

# **Initial Thoughts** Possible classification models include KNN, LDA, QDA, and Logistic Regression. Each classification model performs better under certain conditions.
# 
# Logistic and LDA perform under similar conditions and both draw linear decision boundaries. However, LDA makes the assumption that the observations are normally distributed. That is, for each class, if the features are normally distributed, then an LDA model will perform better. 
# 
# KNN on the other hand is a non parametric model. Thus, it is the most flexible model and will dominate LDA and Logistic models when the decision boundary is highly nonlinear. 
# 
# QDA acts as the middle ground between LDA and KNN because it generates quadratic linear models, and thus allows for slightly more flexible boundaries. 

# **Dataset 2**

# In[511]:

#Load the data, split into train and test
dt_2 = np.loadtxt('datasets/dataset_2.txt')
dt_2train, dt_2test = sk_split(dt_2, train_size = .7)
x_train = dt_2train[:,0:2]
x_test = dt_2test[:,0:2]
y_train = dt_2train[:,-1]
y_test = dt_2test[:,-1]
dt_2_x = dt_2[:,0:2]
dt_2_lat = dt_2[:,0:1]
dt_2_lon = dt_2[:,1:2]
dt_2_y = dt_2[:,-1]


# In[512]:

##Explore the data set by scattering latitude by longitude and color code by class to visualize the shape of the points
#Plot a scatter plot of each class

plt.figure(figsize=(8,8))
high = plt.scatter(dt_2_lat[dt_2_y == 1], dt_2_lon[dt_2_y == 1], c ='g')
low = plt.scatter(dt_2_lat[dt_2_y == 0], dt_2_lon[dt_2_y == 0], c='r')

#Prettify 
plt.xlabel("Latitude")
plt.ylabel('Longitude')
plt.suptitle('Lat - Lon Visualization by class' )
plt.legend((high, low), ("High", "Low"))
plt.show()


# In[513]:

fig,(ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,5))

ax1.hist(dt_2_lat[dt_2_y ==0], bins = 50)
ax1.set_xlabel('Lat')
ax1.set_title('Class: Low/Middle')

ax2.hist(dt_2_lon[dt_2_y ==0], bins = 50)
ax2.set_xlabel('Lon')
ax2.set_title('Class: Low/Middle')

ax3.hist(dt_2_lat[dt_2_y ==1], bins = 50)
ax3.set_xlabel('Lat')
ax3.set_title('Class: High')

ax4.hist(dt_2_lon[dt_2_y ==1], bins = 50)
ax4.set_xlabel('Lon')
ax4.set_title('Class: Low/Middle')
plt.show()


# In[514]:

#Test for the best KNN K value
best_knn2 = []
for i in range(1, 201):
    knn2 = KNN(n_neighbors = i)
    knn2.fit(x_train, y_train)
    knn_score = knn2.score(x_test, y_test)
    best_knn2.append(knn_score)
    
print 'The best K for KNN modeling is ' + str(np.argmax(best_knn2) + 1)


# From the graphs above, we can eliminate KNN because we see that the distributions are about normally distributed. Furthermore, as we can see from the scatter plot, there is overlap between the two groups. As a result, we can judge that KNN will be a poor model. Nonetheless, we will use numerical confirmation and plot its accuracy. 

# In[515]:

#Create all our models
logit2 = LogReg()
logit2.fit(x_train, y_train)
logit2_score = logit2.score(x_test, y_test)
lda2 = da.LinearDiscriminantAnalysis()
lda2.fit(x_train, y_train)
lda2_score = lda2.score(x_test, y_test)
qda2 = da.QuadraticDiscriminantAnalysis()
qda2.fit(x_train, y_train)
qda2_score = qda2.score(x_test, y_test)
knn2 = KNN(n_neighbors = 38)
knn2.fit(x_train, y_train)
knn2_score = knn2.score(x_test, y_test)


# In[516]:

#Plot all models and accuray
fig,(ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
plot_decision_boundary(dt_2_x, dt_2_y, logit2, False, 'Logit: Accuracy = ' + str(logit2_score), ax1)
plot_decision_boundary(dt_2_x, dt_2_y, lda2, False, 'LDA: Accuracy = ' + str(lda2_score), ax2)
plot_decision_boundary(dt_2_x, dt_2_y, qda2, False, 'QDA: Accuracy = ' + str(qda2_score), ax3)
plot_decision_boundary(dt_2_x, dt_2_y, knn2, False, 'KNN: Accuracy = ' + str(knn2_score), ax4)
plt.show()


# **Conclusion** Judging from the histograms, we see that the features are normally distributed, which makes us feel like LDA or QDA may be a better fit. Then, by plotting the data, we see that QDA is the best fit. This is also supported by the numerical data, where the accuracy of the model is highest for QDA. As suspected, KNN proved to be the weakest fit. This follows the reasoning throughout the process. Logistic modeling was not as strong as QDA and LDA because the features follow a normal distribution. On top of that QDA provideda  better fit because the values showed a quadratic boundary. Thus, we can conclude that QDA is the best fit for dataset 2. 

# **Dataset 3**

# In[517]:

dt_3 = np.loadtxt('datasets/dataset_3.txt')
dt_3train, dt_3test = sk_split(dt_3, train_size = .5)
x_train = dt_3train[:,0:2]
x_test = dt_3test[:,0:2]
y_train = dt_3train[:,-1]
y_test = dt_3test[:,-1]
dt_3_x = dt_3[:,0:2]
dt_3_lat = dt_3[:,0:1]
dt_3_lon = dt_3[:,1:2]
dt_3_y = dt_3[:,-1]


# In[518]:

##Explore the data set by scattering latitude by longitude and color code by class to visualize the shape of the points
#Plot a scatter plot of each class
plt.figure(figsize=(6,6))
high = plt.scatter(dt_3_lat[dt_3_y == 1], dt_3_lon[dt_3_y == 1], c ='g')
low = plt.scatter(dt_3_lat[dt_3_y == 0], dt_3_lon[dt_3_y == 0], c='r')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.suptitle('Lat - Long Scatter Visual')
plt.legend((high, low), ("High", "Low"))
plt.show()


# Looking at the data, we see distinct classes. Initially, this has me thinking that KNN would be a much better fit than a logistic model. Furthermore, as seen below, the distributions are not normal, so LDA seems to be poor fits. QDA may be a plausible fit, as it is more flexible than logistic and LDA modeling. Nonetheless, I would predict that KNN will be the best.  To use KNN, we will try to find the best K value. 

# In[519]:

fig,(ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,5))
#fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.hist(dt_3_lat[dt_3_y ==0], bins = 50)
ax1.set_xlabel('Lat')
ax1.set_title('Class: Low/Middle')

ax2.hist(dt_3_lon[dt_3_y ==0], bins = 50)
ax2.set_xlabel('Lon')
ax2.set_title('Class: Low/Middle')

ax3.hist(dt_3_lat[dt_3_y ==1], bins = 50)
ax3.set_xlabel('Lat')
ax3.set_title('Class: High')

ax4.hist(dt_3_lon[dt_3_y ==1], bins = 50)
ax4.set_xlabel('Lon')
ax4.set_title('Class: Low/Middle')
plt.show()


# In[520]:

#Test for the best KNN K value
best_knn = []
for i in range(1, 201):
    knn3 = KNN(n_neighbors = i)
    knn3.fit(x_train, y_train)
    knn_score = knn3.score(x_test, y_test)
    best_knn.append(knn_score)
    
print 'The best K for KNN modeling is ' + str(np.argmax(best_knn) + 1)


# In[521]:

logit3 = LogReg()
logit3.fit(x_train, y_train)
logit3_score = logit3.score(x_test, y_test)
lda3 = da.LinearDiscriminantAnalysis()
lda3.fit(x_train, y_train)
lda3_score = lda3.score(x_test, y_test)
qda3 = da.QuadraticDiscriminantAnalysis()
qda3.fit(x_train, y_train)
qda3_score = qda3.score(x_test, y_test)
knn3 = KNN(n_neighbors = 1)
knn3.fit(x_train, y_train)
knn3_score = knn3.score(x_test, y_test)


# In[522]:

fig,(ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
plot_decision_boundary(dt_3_x, dt_3_y, logit3, False, 'Logit: Accuracy = ' + str(logit2_score), ax1)
plot_decision_boundary(dt_3_x, dt_3_y, lda3, False, 'LDA: Accuracy = ' + str(lda2_score), ax2)
plot_decision_boundary(dt_3_x, dt_3_y, qda3, False, 'QDA: Accuracy = ' + str(qda2_score), ax3)
plot_decision_boundary(dt_3_x, dt_3_y, knn3, False, 'KNN: Accuracy = ' + str(knn3_score), ax4)
plt.show()


# **Conclusion** As postulated, KNN was the best model. This supports the findings from below: there was no clear linear decision boundary, so both Logit and LDA models did not seem plausible. Especially for LDA, the features were not normally distributed, so LDA clearly would not work. QDA seemed like a better fit with more flexible bounds. However, because the decision boundary is highly non-linear, KNN is the best model. This is supported by both the visualizations and the numerical analysis. KNN model had the highest accuracy, followed up QDA and then LDA and Logit

# ---

# ## Challenge Problem: Regularization
# 
# We have seen ways to include different forms of regularizations in Linear regression and Logistic regression, in order to avoid overfitting. We will now explore ways to incorporate regularization within the discriminant analysis framework.
# 
# - When we have a small training sample, we end up with poor estimates of the class proportions $\pi_i$ and covariance matrices $\Sigma$. How can we regularize these quantities to improve the quality of the fitted model?
# 
# 
# - We have seen that different assumptions on the covariance matrix results in either a linear or quadratic decision boundary. While the former may yield poor prediction accuracy, the latter could lead to over-fitting. Can you think of a suitable way to regularize the covariance to have an intermediate fit?
# 
# The solutions that you suggest must include a parameter that allows us to control the amount of regularization.
# 
# Be detailed in your explanation and support your reasoning fully. You do not, however, need to implement any of these solutions.

# **Solution** 
# When using small training samples, the covariance matrix estimates become highly variable. As a result, smaller egienvalues of the variance matrix are more heavily weighted than the larger eigenvalues. To fix this, we can use a regularization method, which is taken in as a paramter when we use discriminant anlysis with sklearn. By using a regularization parameter, we see a trade off between the variance of our covariance matrix and bias. In general, we are trying to use our observations to estimate the population, so whether we are striving for large or little variance depends on how confident we are that our observations match the population. If our observations closely match the population, then we want to use a high degree of regularization to  minimize our variance, at the cost of bias. We would do the opposite if we believed that our observations did not match the true population.
# 
# When deciding beteween QDA and LDA, we have a regualrization paramter that is utilized for our covariance matrix. A lambda value of 0 would result in QDA and a lambda value of 1 would result in LDA. Thus, to have an intermediate fit, we would use a regualrizaton parameter between 0 and 1.
# 
# This problem was solved using resources from http://slac.stanford.edu/cgi-wrap/getdoc/slac-pub-4389.pdf

# In[ ]:



