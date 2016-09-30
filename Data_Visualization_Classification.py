###Data Visualization and Classification

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
%matplotlib inline


###Read the dataset and use some basic steps of data visualization and statistical analysis
#read data into pandas df
df = pd.read_csv('dataset_HW1.txt')

#size of data frame
print 'number of patients:', df.shape[0]

#print first 5 rows of dataframe
df.head(n=5)

#choose columns 1, 2, 3, 4 (ignoring columns 0 and 4)
df_cols_1_to_4 = df[range(1, 5)] 

#get column names from important_column dataframe
column_names = df_cols_1_to_4.columns.values 

#create pandas dataframe with column names given by column_names
stats = pd.DataFrame(columns=column_names)

#create a row called 'max' and store max values from the columns of important_columns
stats.loc['max'] = df_cols_1_to_4.max()

#create a row called 'min' and store min values from the columns of important_columns
stats.loc['min'] = df_cols_1_to_4.min()

#create a row called 'range' and store range of values from the columns of important_columns
stats.loc['range'] = df_cols_1_to_4.max() - df_cols_1_to_4.min()
stats.head(n=5)

#choose columns 1, 2, 4 (ignoring columns 0, 3 and 4)
df_cols_1_2_4 = df[[1, 2, 4]] 

stats.loc['mean'] = df_cols_1_2_4.mean()
stats.loc['median'] = df_cols_1_2_4.median()
stats.loc['std'] = df_cols_1_2_4.std()
stats



###Find the mean, median, and std deviation of the three subgroups: children, adult women, and adult men and give a summary of the information
#Create three different datasets by filtering our entire set
children_data = df[df['patient_age'] < 18]
adult_women_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'female')]
adult_male_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'male')]
    
#create dataframe with select column names (just like before)
column_names = children_data[range(1, 5)].columns.values
child_stats = pd.DataFrame(columns=column_names)

#add a row for each stat (just like before)
child_stats.loc['child_max'] = children_data[range(1, 5)].max()
child_stats.loc['child_min'] = children_data[range(1, 5)].min()
child_stats.loc['child_range'] = children_data[range(1, 5)].max() - children_data[range(1, 5)].min()

child_stats.loc['child_mean'] = children_data[[1, 2, 4]].mean()
child_stats.loc['child_median'] = children_data[[1, 2, 4]].median()
child_stats.loc['child_std'] = children_data[[1, 2, 4]].std()

child_stats

#create dataframe with select column names (just like before)
column_names = adult_women_data[range(1, 5)].columns.values
adult_women_stats = pd.DataFrame(columns=column_names)

#add a row for each stat (just like before)
adult_women_stats.loc['adult_f_max'] = adult_women_data[range(1, 5)].max()
adult_women_stats.loc['adult_f_min'] = adult_women_data[range(1, 5)].min()
adult_women_stats.loc['adult_f_range'] = adult_women_data[range(1, 5)].max() - adult_women_data[range(1, 5)].min()

adult_women_stats.loc['adult_f_mean'] = adult_women_data[[1, 2, 4]].mean()
adult_women_stats.loc['adult_f_median'] = adult_women_data[[1, 2, 4]].median()
adult_women_stats.loc['adult_f_std'] = adult_women_data[[1, 2, 4]].std()
adult_women_stats

#create dataframe with select column names (just like before)
column_names = adult_male_data[range(1, 5)].columns.values
adult_male_stats = pd.DataFrame(columns=column_names)

#add a row for each stat (just like before)
adult_male_stats.loc['adult_m_max'] = adult_male_data[range(1, 5)].max()
adult_male_stats.loc['adult_m_min'] = adult_male_data[range(1, 5)].min()
adult_male_stats.loc['adult_m_range'] = adult_male_data[range(1, 5)].max() - adult_male_data[range(1, 5)].min()

adult_male_stats.loc['adult_m_mean'] = adult_male_data[[1, 2, 4]].mean()
adult_male_stats.loc['adult_m_median'] = adult_male_data[[1, 2, 4]].median()
adult_male_stats.loc['adult_m_std'] = adult_male_data[[1, 2, 4]].std()
adult_male_stats

#children vs adults pie chart
children = children_data.shape[0] #number of kids
adults = df.shape[0] - children #adults = total number - kids

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(221)
ax1.pie([children, adults], 
        labels=['children: ' + str(children), 'adults: ' + str(adults)], 
        colors=['red', 'gold'],
        autopct='%1.1f%%', shadow=True, startangle=90)

#women vs men pie chart
women = df[df['patient_gender'] == 'female'].shape[0] #number of women
men = df.shape[0] - women #men = total number - women

ax2 = fig.add_subplot(222)
ax2.pie([women, men], 
        labels=['women: ' + str(women), 'men: ' + str(men)], 
        colors=['lightskyblue', 'yellowgreen'],
        autopct='%1.1f%%', shadow=True, startangle=90)

#adult women vs men pie chart
adult_women = adult_women_data.shape[0]
adult_men = adults - adult_women

ax3 = fig.add_subplot(223)
ax3.pie([adult_women, adult_men], 
        labels=['adult women: ' + str(adult_women), 'adult men: ' + str(adult_men)], 
        colors=['pink', 'lightblue'],
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.show()

#function for plotting histograms
def plot_hist(data, title, x_label, face, axes):
    
    axes.hist(data, 
         50, 
         normed=1, 
         facecolor=face, 
         alpha=0.75)
    
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel('frequency')
    
    return axes
    


###Plot histograms describing marker values and scatter plots with different colors for the different subtypes
#plot histograms for each marker and each demographics
#in the following, instead of adding one subplot to a 4x2 grid at a time
#I can get all the subplot axes for the grid in one line 
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10, 10))
ax1 = plot_hist(df['marker_1'],
                'Histogram of marker 1', 
                'marker 1 value', 
                'green', 
                ax1)

ax2 = plot_hist(df['marker_2'],
                'Histogram of marker 2', 
                'marker 2 value', 
                'red', 
                ax2)

ax3 = plot_hist(children_data['marker_1'],
                'Histogram of marker 1 for children', 
                'marker 1 value', 
                'green', 
                ax3)

ax4 = plot_hist(children_data['marker_2'],
                'Histogram of marker 2 for children', 
                'marker 2 value', 
                'red', 
                ax4)

ax5 = plot_hist(adult_women_data['marker_1'],
                'Histogram of marker 1 for adult women', 
                'marker 1 value', 
                'green', 
                ax5)

ax6 = plot_hist(adult_women_data['marker_2'],
                'Histogram of marker 2 for adult women', 
                'marker 2 value', 
                'red', 
                ax6)

ax7 = plot_hist(adult_male_data['marker_1'],
                'Histogram of marker 1 for adult men', 
                'marker 1 value', 
                'green', 
                ax7)

ax8 = plot_hist(adult_male_data['marker_2'],
                'Histogram of marker 2 for adult men', 
                'marker 2 value', 
                'red', 
                ax8)

plt.tight_layout()
plt.show()

def plot_scatter(data, plot_title, x_lable, y_lable, groups, axes):
    #set up color map (one color per group number)
    
    #split [0, 1] in to as many parts as there are groups
    group_numbers = np.linspace(0, 1, groups) 
    #get a color map
    c_map = plt.get_cmap('rainbow') 
    #get a range of colors from color map
    c_norm  = colors.Normalize(vmin=0, vmax=group_numbers[-1])
    #get a map that maps a group number to a color
    number_map = cmx.ScalarMappable(norm=c_norm, cmap=c_map)
    
    #plot points colored by their group number
    for group in xrange(groups):
        #convert a group number into a color using our map
        color = number_map.to_rgba(group_numbers[group])
        #make a scatter plot of a specific group colored by its group number color
        axes.scatter(data[data['subtype'] == group]['marker_1'], 
                     data[data['subtype'] == group]['marker_2'], 
                     c=color, 
                     alpha = 0.5)

    axes.set_title(plot_title) 
    axes.set_xlabel(x_lable)
    axes.set_ylabel(y_lable)
    
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

plot_scatter(df, 'scatter plot of entire dataset', 'marker 1', 'marker 2', 4, ax1)  
plot_scatter(children_data, 
             'scatter plot of children', 
             'marker 1', 'marker 2', 
             4,
             ax2)  
plot_scatter(adult_women_data, 
             'scatter plot of adult female', 
             'marker 1', 
             'marker 2', 
             4, 
             ax3)  
plot_scatter(adult_male_data, 
             'scatter plot of adult male', 
             'marker 1', 
             'marker 2', 
             4, 
             ax4) 

plt.tight_layout()
plt.show()


###Problem 2
#Use information from part 1 to classify the disease subtype of new patients.
#Randomly split data in a training and testing data set
#Write a function that computes biometric mean for each subtype
#Classify disease subtype based off of which mean it is most similar to (similarity is based off Euclidean distance)
#Evaluate your results by finding the percentage of correct classifications

#Create a distance function for classifying
def distance (x, y):
    dist = np.sqrt((x[1] - y[1])**2 + (x[0] - y[0])**2)
    return dist

#Create the classify function
def classify(training, testing):
    #Find the mean marker values for each of the training group subtypes
    testing['predicted'] = ""
    predicted_list = []
    training_subtype0 = (training[training['subtype']==0]['marker_1'].mean(),training[training['subtype']==0]['marker_2'].mean())
    training_subtype1 = (training[training['subtype']==1]['marker_1'].mean(), training[training['subtype']==1]['marker_2'].mean())
    training_subtype2 = (training[training['subtype']==2]['marker_1'].mean(), training[training['subtype']==2]['marker_2'].mean())
    training_subtype3 = (training[training['subtype']==3]['marker_1'].mean(), training[training['subtype']==3]['marker_2'].mean())
    
    #Loop through each row in testing, while finding which subtype is it closest too
    #Add that value into predicated subtype list and then add list to the dataframe
    for index, row in testing.iterrows():
        testing_markers = (row['marker_1'], row['marker_2'])
        distance_data = [distance(training_subtype0, testing_markers), distance(training_subtype1, testing_markers), distance(training_subtype2, testing_markers), distance(training_subtype3, testing_markers)]
        distance_min = min(distance_data)
        predicted_subtype = distance_data.index(distance_min)
        predicted_list.append(predicted_subtype)

    testing['predicted'] = predicted_list        
    return testing

#Create evaluating function
def evaluate (testing):    
    #Create counters for correct and incorrect answers
    correct = 0
    incorrect = 0
    
    #Loop through all the rows; if the subtype and predicted subtype match, the classified work and add onto correct
    for index, row in testing.iterrows():
        if int(row['subtype'] == row['predicted']):
            correct = correct + 1            
        else:
            incorrect = incorrect + 1
    #Calculate and return accuracy
    accuracy = float(correct)/(incorrect + correct)
    return accuracy

#Execute and evaluate with children training data
#Randomly split the child patients into two groups
training_children = children_data.sample(frac=.7, replace=False)
testing_children = children_data.drop(training_children.index)

#Classify and then evaulate the two random groups
classify(training_children, testing_children)
child_percent = evaluate(testing_children)*100
print "The classification system was " + str(child_percent) + "% accurate"


#Repeat with both the women and male data
#Randomly split the adult female patients into the two groups
training_adult_female = adult_women_data.sample(frac = .7)
testing_adult_female = adult_women_data.drop(training_adult_female.index)

#Classify and then evaulate the two random groups
classify(training_adult_female, testing_adult_female)
adult_female_percent = evaluate(testing_adult_female)*100
print "The classification system was " + str(adult_female_percent) + "% accurate"

#Randomly split the adult male patients into the two groups
training_adult_male = adult_male_data.sample(frac=.7, replace=False)
testing_adult_male = adult_male_data.drop(training_adult_male.index)

#Classify and then evaulate the two random groups
classify(training_adult_male, testing_adult_male)
adult_male_percent = evaluate(testing_adult_male)*100
print "The classification system was " + str(adult_male_percent) + "% accurate"



###Probalm 3
#create a new classifying system where we classify the subtype of an unknown as the subtype of the most similar training point
#Classify_2 is named so to avoid confusion with problem 2
def classify_2(training, testing):
    testing['predicted'] = ""
    prediction_list = []
    
    #Loop through testing
    for index, testing_row in testing.iterrows():
        
        #Initialize distance to some huge number, index to keep track of where the min distance is, and the markers
        dist = 9999999
        min_subtype = 0
        training_markers = (0,0)
        testing_markers = (testing_row['marker_1'], testing_row['marker_2'])
        
        #Loop through each of the rows in training and calculate the distance between each training marker value
        #Find the row with  minimum distance and store the respective subtype in a list 
        for index, training_row in training.iterrows():
            training_markers = (training_row['marker_1'], training_row['marker_2'])
            new_dist = distance(training_markers, testing_markers)
            #Test to see if the distance of a row is less than the shortest row found thus far; if so save subtype and distance
            if (dist > new_dist):
                min_subtype = training_row['subtype']
                dist = new_dist
        prediction_list.append(min_subtype)
    testing['predicted'] = prediction_list
    return testing


#Randomly split the child patients into two groups
training_children = children_data.sample(frac=.7, replace=False)
testing_children = children_data.drop(training_children.index)

#Classify and then evaulate the two random groups
classify_2(training_children, testing_children)
child_percent = evaluate(testing_children)*100
print "The classification system was " + str(child_percent) + "% accurate"

#Randomly split the adult female patients into the two groups
training_adult_female = adult_women_data.sample(frac = .7)
testing_adult_female = adult_women_data.drop(training_adult_female.index)

#Classify and then evaulate the two random groups
classify_2(training_adult_female, testing_adult_female)
adult_female_percent = evaluate(testing_adult_female)*100
print "The classification system was " + str(adult_female_percent) + "% accurate"

#Randomly split the adult male patients into the two groups
training_adult_male = adult_male_data.sample(frac=.7, replace=False)
testing_adult_male = adult_male_data.drop(training_adult_male.index)

#Classify and then evaulate the two random groups
classify_2(training_adult_male, testing_adult_male)
adult_male_percent = evaluate(testing_adult_male)*100
print "The classification system was " + str(adult_male_percent) + "% accurate"


###Challenge Problem
#Download a table of voting data from US Census Bureau

#Import the excel file 
voting_df = pd.read_excel('PSET1_Q4.xls', header = None)

#Create column names
column_names = ['Sex and Age', 'Population', 'Citizen population', 'Reported registered #', 'Reported registered %', 'Reported not registered #', 'Reported not registered %', "No response to registration #", "No response to registration %", "Reported voted #", "Reported voted %", "Reported no vote #", "Reported no vote %", "No response to voting #", "No response to voting %", "Reported registered total %", "Reported voted total %"]

#Create a new df that splices off the extra space at the top of the excel import
data_df = voting_df.loc[5,]

#Perform data exploration
#Create a dataframe that contains the sex and age, population and citizen population
#Drop blank values

all_gender = voting_df.loc[9:81,]
all_gender.columns = column_names
all_gender = all_gender.drop(voting_df.index[[17]])
all_gender.append(data_df)

#Create two dataframes that capture the total population and the total citizens population
total_population = all_gender.loc[9, 'Population']
citizens = all_gender.loc[9, "Citizen population"]

#Create a female data frame
female_data = voting_df.loc[161:233,]
female_data.columns = column_names
female_data = female_data.drop(df.index[[169]])
total_population_female = female_data.loc[161, 'Population']
female_citizens = female_data.loc[161, "Citizen population"]

#Create a male dataframe and find the value of total male and male citizen population
male_data = voting_df.loc[85:157,]
male_data.columns = column_names
male_data = male_data.drop(df.index[[93]])
total_population_male = male_data.loc[85, 'Population']
male_citizens = male_data.loc[85, "Citizen population"]

#Find number of citizens and non citizens for overall, males, and females
not_citizens = total_population - citizens
male_not_citizens = total_population_male - male_citizens
female_not_citizens = total_population_female - female_citizens

Create 4 pie graphs, displaying the sex breakdown of population, breakdown of citizens and non citizens
#and citizen breakdowns of males and females, and breakdown

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(221)
ax1.pie([total_population_male, total_population_female], 
        labels=['Male: ' + str(total_population_male),  ' Female ' + str(total_population_female)], 
        colors=['lightblue', 'pink'],
        autopct='%1.1f%%', shadow=True, startangle=90)

ax2 = fig.add_subplot(222)
ax2.pie([citizens, not_citizens], 
        labels=['Citizens: ' + str(citizens),  ' Not Citizens ' + str(not_citizens)], 
        colors=['blue', 'red'],
        autopct='%1.1f%%', shadow=True, startangle=90)

ax3 = fig.add_subplot(223)
ax3.pie([citizens_female, noncitizens_female], 
        labels=['Female Citizens: ' + str(citizens_female), 'Female Not Citizens: ' + str(female_not_citizens)], 
        colors=['yellow', 'lightgreen'],
        autopct='%1.1f%%', shadow=True, startangle=90)

ax4 = fig.add_subplot(224)
ax4.pie([citizens_male, noncitizens_male], 
        labels=['Male Citizens: ' + str(male_citizens), 'Male Not Citizens: ' + str(male_not_citizens)], 
        colors=['salmon', 'purple'],
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.show()

#Break up the all gender dataframe, male dataframe, and female dataframe into raw numbers
all_gender_number = all_gender[['Reported registered #', 'Reported not registered #', "No response to registration #", "Reported voted #", "Reported no vote #", "No response to voting #"]]
male_number = male_data[['Reported registered #', 'Reported not registered #', "No response to registration #", "Reported voted #", "Reported no vote #", "No response to voting #"]]
female_number = female_data[['Reported registered #', 'Reported not registered #', "No response to registration #", "Reported voted #", "Reported no vote #", "No response to voting #"]]
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

#Pull values from all gender, males, and females
all_gender_total = np.array(all_gender_number.loc[9,])
male_number_total = np.array(male_number.loc[85,])
female_number_total = np.array(female_number.loc[161,])

#Create an array of six to be used when making the bar; make width of .25
category_number = np.arange(6)
bar_width = 0.25


#Create bars for the entire population, males, and then females
bar1 = ax.bar(category_number, all_gender_total, width,color='yellow')
bar2 = ax.bar(category_number + bar_width, male_number_total, width,color='pink')
bar3 = ax.bar(category_number + (2 * bar_width), female_total, width,color='lightblue')

#Create labels of graph
ax.set_ylabel('Number of People')
ax.set_xticks(category_number + width)
ax.set_xticklabels(('Registered', 'Not Registered.', "Reg. No Response", "Voted", "No vote", "Vote No Response"))
ax.legend((bar1[0], bar2[0], bar3[0]), ("All Genders", "Males", "Females"))
plt.xticks(fontsize = 9)
plt.title("Voting Statistics Breakdown by Gender")

plt.show()

#Break up the three groups by percentage for 8 categories and create a graph
all_gender_percent = all_gender[['Reported registered %', 'Reported not registered %', "No response to registration %", "Reported voted %", "Reported no vote %", "No response to voting %", "Reported registered total %", "Reported voted total %"]]
male_percent = male_data[['Reported registered %', 'Reported not registered %', "No response to registration %", "Reported voted %", "Reported no vote %", "No response to voting %", "Reported registered total %", "Reported voted total %"]]
female_percent = female_data[['Reported registered %', 'Reported not registered %', "No response to registration %", "Reported voted %", "Reported no vote %", "No response to voting %", "Reported registered total %", "Reported voted total %"]]
fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111)

all_gender_percent_total = np.array(both_per.loc[9,])
male_percent_total = np.array(male_per.loc[85,])
female_percent_total = np.array(female_per.loc[161,])

category_percent = np.arange(8)

width = 0.25

#Create the three bars
bar1 = ax.bar(category_percent, all_gender_percent_total[0:8], width,color='yellow')
bar2 = ax.bar(category_percent + width, male_percent_total[0:8], width,color='lightblue')
bar3 = ax.bar(category_percent + (2 * width), female_percent_total[0:8], width,color='pink')

#Create labels and size
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of people')
ax.set_xticks(category_percent + width)
ax.set_xticklabels(('Reg', 'Not Reg', "Reg. No Response", "Voted", "No Vote", "No Response Vote","Rep. Reg.", "Rep Vote"))
ax.legend((bar1[0], bar2[0], bar3[0]), ("All genders", "Males", "Females"))
plt.xticks(fontsize = 9)
plt.title("Voting Statistics Percentage Breakdown by Gender")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
age_groups_data = all_gender.loc[10:17,]
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create registered and nonregistered, which is the statistic we are looking for
registered = np.array(age_groups_data['Reported registered %'])
not_registered = np.array(age_groups_data['Reported not registered %'])
no_response = np.array(age_groups_data['No response to registration %'])


#Create two bars, one for registered and the other for not registered 
bar1 = ax.bar(categories, registered, width, color='blue')
bar2 = ax.bar(categories + width, not_registered, width, color='red')
bar3 = ax.bar(categories + 2*width, no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ("Registered", "Not Registered", "No Response"))
plt.title("Registration Percentage Breakdown by Age")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create voting, no voting, and no response, which is the statistic we are looking for
voting = np.array(age_groups_data['Reported voted %'])
not_voting = np.array(age_groups_data['Reported no vote %'])
no_response = np.array(registered_groups['No response to voting %'])

#Create three bar
bar1 = ax.bar(categories, voting, width, color='blue')
bar2 = ax.bar(categories + width, not_voting, width, color='red')
bar3 = ax.bar(categories + (2 * width), no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ('Voting', 'Not Voting', 'No Response'))
plt.title("Voting Percentage Breakdown by Age")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
male_bar_data = male_percent.loc[86:93,]
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create registered and nonregistered, which is the statistic we are looking for; input is males
registered = np.array(male_bar_data['Reported registered %'])
not_registered = np.array(male_bar_data['Reported not registered %'])
no_response = np.array(male_bar_data['No response to registration %'])

#Create two bars, one for registered and the other for not registered 
bar1 = ax.bar(categories, registered, width, color='blue')
bar2 = ax.bar(categories + width, not_registered, width, color='red')
bar3 = ax.bar(categories + (2 * width), no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ("Registered", "Not Registered", "No Response"))
plt.title("Male Registration Percentage Breakdown by Age")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create voting, no voting, and no response, which is the statistic we are looking for
voting = np.array(male_bar_data['Reported voted %'])
not_voting = np.array(male_bar_data['Reported no vote %'])
no_response = np.array(male_bar_data['No response to voting %'])

#Create three bars, one for each response type
bar1 = ax.bar(categories, voting, width, color='blue')
bar2 = ax.bar(categories + width, not_voting, width, color='red')
bar3 = ax.bar(categories + (2 * width), no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ('Voting', 'Not Voting', 'No Response'))
plt.title("Male Voting Percentage Breakdown by Age")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
female_bar_data = female_percent.loc[162:169,]
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create registered and nonregistered, which is the statistic we are looking for; input is males
registered = np.array(female_bar_data['Reported registered %'])
not_registered = np.array(female_bar_data['Reported not registered %'])
no_response = np.array(female_bar_data['No response to registration %'])

#Create two bars, one for registered and the other for not registered 
bar1 = ax.bar(categories, registered, width, color='blue')
bar2 = ax.bar(categories + width, not_registered, width, color='red')
bar3 = ax.bar(categories + (2 * width), no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ("Registered", "Not Registered","No Response"))
plt.title("Femle Registration Percentage Breakdown by Age")

plt.show()

#Create a dataframe using the age groupings provided in the spreadsheet and create the graph
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
categories = np.arange(7)
width = 0.20

#Create voting, no voting, and no response, which is the statistic we are looking for
voting = np.array(female_bar_data['Reported voted %'])
not_voting = np.array(female_bar_data['Reported no vote %'])
no_response = np.array(female_bar_data['No response to voting %'])

#Create three bars, one for each response type
bar1 = ax.bar(categories, voting, width, color='blue')
bar2 = ax.bar(categories + width, not_voting, width, color='red')
bar3 = ax.bar(categories + (2 * width), no_response, width, color='black')

#Create labels
ax.set_ylim(0,100)
ax.set_ylabel('Percentage of populations')
ax.set_xticklabels(("18-24 years", "25-34 years", "35-44 years", "45-54 years", "55-64 years", "65-74 years", '75 years and over'))
plt.xticks(fontsize = 9)
ax.set_xticks(categories + width)
ax.legend((bar1[0], bar2[0], bar3[0]), ('Voting', 'Not Voting', 'No Response'))
plt.title("Female Voting Percentage Breakdown by Age")

plt.show()
