#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[2]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer,StandardScaler
import time

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv("Udacity_AZDIAS_Subset.csv",sep=";")

# Load in the feature summary file.
feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv",sep=";")


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

#count of rows and columns
print("Azdias shape:",azdias.shape)
print("Preview Azdias:",azdias.head(3))
print("Column types:",azdias.dtypes)
display (azdias.describe ())


# In[36]:



#print("feat_info shape:",feat_info.shape)

print("Preview feat_info:",feat_info.head(3))

print("Column types:",feat_info.dtypes)


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[4]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

# Identify missing or unknown data values and convert them to NaNs.
null_percent= (azdias.isnull().sum()/891221)*100
display(null_percent.sort_values(ascending = False))
plt.hist(null_percent)


# In[38]:


# Investigate patterns in the amount of missing data in each column.
def checkint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

data_dict = {'nan_vals': feat_info['missing_or_unknown'].str.replace('[','').str.replace(']','').str.split(',').values}

missing_vals = pd.DataFrame(data_dict, index = feat_info['attribute'].values)
missing_vals['nan_vals'] = missing_vals.apply(lambda x: [int(i) if checkint(i) == True else i for i in x[0]], axis=1)


# In[7]:


azdias['CAMEO_DEUG_2015'][2511] # run before and after next code to check if mapping is correct


# In[39]:


for column in azdias.columns:
    # Get index 0 of missing_vals.loc[column] to get actual array
    azdias[column] = azdias[column].replace(missing_vals.loc[column][0], np.nan)


# In[40]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

null_percent= (azdias.isnull().sum()/891221)*100

plt.hist(null_percent)


# In[41]:


#identify columns with more than 20% missing values
null_percent[null_percent > 20]


# In[42]:


azdias = azdias.drop(['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP','KBA05_BAUMAX'], axis=1)


# In[43]:



feat_info = feat_info[~feat_info.attribute.isin(['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP','KBA05_BAUMAX'])]


# In[44]:


azdias.columns


# In[15]:


feat_info


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# I have removed columns with more than 20% missing values as they appear to be outliers (based on the histogram).
# Following columns are removed :
# AGER_TYP        76.955435
# GEBURTSJAHR     44.020282
# TITEL_KZ        99.757636
# ALTER_HH        34.813699
# KK_KUNDENTYP    65.596749
# KBA05_BAUMAX    53.468668

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[14]:


# How much data is missing in each row of the dataset?
#histogram of null values across rows
plt.hist(azdias.isnull().sum(axis=1),bins = 30,facecolor = 'black')
plt.xlabel('missing values')
plt.ylabel('row count')


# In[45]:


missing_row_count = azdias.isnull().sum(axis=1)

missing_above_20 = missing_row_count[missing_row_count > 20]

missing_below_20 = missing_row_count[missing_row_count <= 20]


# In[46]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
      
#identify columns with zero missing values
zero_missing_columns = null_percent[null_percent==0].index.tolist()
print("There are",len(zero_missing_columns)," columns with missing elements")

#select subset for comparison, say around 8 fields
for_comparison = zero_missing_columns

def plot_compare(column):
    fig = plt.figure(figsize=(14,4))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Many missing rows')
    sns.countplot(azdias.loc[missing_above_20.index,column])

    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Few missing rows')
    sns.countplot(azdias.loc[missing_below_20.index,column]);

    fig.suptitle(column)
    plt.show()

for i in range(len(for_comparison)):
    plot_compare(for_comparison[i])
#plot_compare('FINANZ_HAUSBAUER')
#plot_compare('LP_FAMILIE_FEIN')
#plot_compare('FINANZ_VORSORGER')
#plot_compare('GREEN_AVANTGARDE')


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# From the plots , it appears that there is a difference in the quality of data . All features except "ANDREDE_KZ" have a substantial difference in distribution of values. Because of this, rows with more than 20 missing values have been removed.
# 

# In[47]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
# dividing the data into 2 categories , above and below 20 missing values
missing_row_count = azdias.isnull().sum(axis=1)

missing_above_20 = missing_row_count[missing_row_count >20]
display('Count of Rows with more than 20 Missing values: ',missing_above_20.shape[0])

missing_below_20 = missing_row_count[missing_row_count <=20]
display('Count of Rows with less than 20 Missing values: ',missing_below_20.shape[0])


# In[48]:


print('rows before:', azdias.shape[0])

azdias= azdias[azdias.index.isin(missing_below_20.index)]

print('rows after:', azdias.shape[0])


# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[13]:


# How many features are there of each data type?


#feat_info
print("Feature distribution by category: ")
feat_info.groupby('type').count()[['attribute']]


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[49]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

categorical_variables = feat_info[feat_info.type == 'categorical'].attribute
categorical_variables


# In[21]:


# Split categorical variables into binary or multi buckets
binary = []
multi = []
for var in categorical_variables:
    if azdias[var].nunique() > 2:
        multi.append(var)
    else:
        binary.append(var)
        
        
print("Multi:", multi)
print("Binary:",binary)


# In[50]:


azdias['CAMEO_DEU_2015'].unique()


# In[22]:


# Re-encode categorical variable(s) to be kept in the analysis.
for b in binary:
    print(azdias[b].value_counts())


# In[53]:


for m in multi:
    print(azdias[m].value_counts())


# In[51]:


#Variable 'OST_WEST_KZ' has to be re-encoded; W:1 O:0
azdias['OST_WEST_KZ'].unique()
azdias['OST_WEST_KZ'] = azdias['OST_WEST_KZ'].map({'W': 1, 'O': 0})


# In[25]:


#azdias['CAMEO_DEU_2015_6B'].unique()


# In[26]:


#Check the mapping
azdias['OST_WEST_KZ'].unique()


# In[52]:


#Encode multi level variables
azdias = pd.get_dummies(azdias, columns=multi)
azdias.columns


# In[54]:


azdias['WOHNLAGE'].unique()
#azdias['PLZ8_BAUMAX'].unique()


# In[55]:



azdias['WOHNLAGE_RURAL'] =  [1 if x in [1.,2.,3.,4.,5.] else 0 for x in azdias['WOHNLAGE']] 


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# The variable 'OST_WEST_KZ' was encoded to binary classification.
# All other binary variables were kept as is.
# One-hot encoding was applied on multi level categorical variables.
# No variables were dropped.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[56]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

azdias['PRAEGENDE_JUGENDJAHRE'].value_counts()

#creating new features Mainstream and Decade
azdias['Mainstream'] = [1 if x in [1,3,5,8,10,12,14] else 0 for x in azdias['PRAEGENDE_JUGENDJAHRE']] 
azdias['Decade'] = azdias['PRAEGENDE_JUGENDJAHRE'].map({1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6})


# In[57]:


#This block is to test if new columns were created properly

print(azdias['Mainstream'].value_counts())
print(azdias['Decade'].value_counts())

print(azdias.groupby(['Mainstream', 'PRAEGENDE_JUGENDJAHRE']).size().reset_index(name='Freq'))
print(azdias.groupby(['Decade', 'PRAEGENDE_JUGENDJAHRE']).size().reset_index(name='Freq'))


# In[22]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.

print(azdias['CAMEO_INTL_2015'].value_counts())


# In[59]:


def wealth(x):
    if pd.isnull(x):
        return np.nan
    else:
        return int(str(x)[0])


def life(x):
    if pd.isnull(x):
        return np.nan
    else:
        return int(str(x)[1])


# In[60]:


azdias['WEALTH_STAGE']  = azdias['CAMEO_INTL_2015'].apply(wealth)
azdias['LIFE_STAGE']  = azdias['CAMEO_INTL_2015'].apply(life)


# In[61]:


print("Check wealth feature :")
print(azdias.groupby(['WEALTH_STAGE', 'CAMEO_INTL_2015']).size().reset_index(name='Freq'))
print("Check life stage feature:")
print(azdias.groupby(['LIFE_STAGE', 'CAMEO_INTL_2015']).size().reset_index(name='Freq'))


# In[62]:


# Remove CAMEO_INTL_2015 and  PRAEGENDE_JUGENDJAHRE
azdias = azdias.drop(['CAMEO_INTL_2015'],axis=1)
azdias = azdias.drop(['PRAEGENDE_JUGENDJAHRE'],axis=1)
azdias = azdias.drop(['WOHNLAGE'],axis=1)
azdias = azdias.drop(['PLZ8_BAUMAX'],axis=1)

azdias.head(5)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# The column PRAEGENDE_JUGENDJAHRE was split into 2 new columns Mainstream and Decade.
# The column CAMEO_INTL_2015 was split into Wealth_Stage and Life_Stage .
# The column WOHNLAGE was recoded as 1 for rural and 0 for non_rural.
# The original columns were then dropped along with PLZ8_BAUMAX as this variable is similar to KBA05_BAUMAX

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[67]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)
azdias.shape
np.unique(azdias.dtypes.values)


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[3]:


def checkint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def wealth(x):
    if pd.isnull(x):
        return np.nan
    else:
        return int(str(x)[0])


def life(x):
    if pd.isnull(x):
        return np.nan
    else:
        return int(str(x)[1])
    
decade_dict = {0: [1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8, 9], 4: [10, 11, 12, 13], 5:[14, 15]}
def map_decade_dict(x):
    for key, value in decade_dict.items():
        if x in value: return key

movement_dict = {0: [1, 3, 5, 8, 10, 12, 14], 1: [2, 4, 6, 7, 9, 11, 13, 15]}

def map_movement_dict(x):
    for key, value in movement_dict.items():
        if x in value: return key

wealthy_dict = {0: [11, 12, 13, 14, 15],
                    1: [21, 22, 23, 24, 25],
                    2: [31, 32, 33, 34, 35],
                    3: [41, 42, 43, 44, 45],
                    4: [51, 52, 53, 54, 55]}
def map_wealthy_dict(x):
    for key, value in wealthy_dict.items():
        if int(x) in value: return key
        
life_stage_dict = {0: [11, 21, 31, 41, 51],
                   1: [12, 22, 32, 42, 52],
                   2: [13, 23, 33, 43, 53],
                   3: [14, 24, 34, 44, 54],
                   4: [15, 25, 35, 45, 55]}

def map_life_stage_dict(x):
    for key, value in life_stage_dict.items():
        if int(x) in value: return key
    
def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    df_info = pd.read_csv("AZDIAS_Feature_Summary.csv",sep=";")
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    df_features = list(df.columns)
    i = 0
    for element in df_features:
        mylist = df_info.loc[i].missing_or_unknown[1:-1].split(',')
        helper_df = df[df_features[i]].isin(mylist)
        df[element] = df[element].where(helper_df == False)
        i=i+1
    
    # remove selected columns and rows, ... 
    drop_columns = ['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX','GEBURTSJAHR','ALTER_HH','CAMEO_DEU_2015']
    df.drop(drop_columns, axis = 1, inplace = True)
    
    # build df with no missing values
    df['helper'] = df.isnull().sum(axis=1)
    df_no_missing_values = df[df['helper']==0]
    df_no_missing_values = df_no_missing_values.drop(['helper'], axis = 1)
 
    # select, re-encode, and engineer column values.
    df_no_missing_values['OST_WEST_KZ'] = df_no_missing_values['OST_WEST_KZ'].map({'W': 1, 'O': 0})
    df_no_missing_values = df_no_missing_values.drop(['LP_FAMILIE_GROB','LP_STATUS_GROB','GEBAEUDETYP'],axis=1)
    df_no_missing_values['WOHNLAGE_RURAL'] =  [1 if x in [1.,2.,3.,4.,5.] else 0 for x in df_no_missing_values['WOHNLAGE']] 
    df_no_missing_values = df_no_missing_values.drop(['WOHNLAGE'], axis=1)
    
    multi_level = ['CJT_GESAMTTYP','FINANZTYP','GFK_URLAUBERTYP','LP_FAMILIE_FEIN','LP_STATUS_FEIN','NATIONALITAET_KZ','SHOPPER_TYP','ZABEOTYP','CAMEO_DEUG_2015']
    df_no_missing_values = pd.get_dummies(df_no_missing_values, columns=multi_level)

    df_no_missing_values['PRAEGENDE_JUGENDJAHRE_decade'] = df_no_missing_values['PRAEGENDE_JUGENDJAHRE'].apply(map_decade_dict)
    df_no_missing_values['PRAEGENDE_JUGENDJAHRE_movement'] = df_no_missing_values['PRAEGENDE_JUGENDJAHRE'].apply(map_movement_dict)
    df_no_missing_values = df_no_missing_values.drop(['PRAEGENDE_JUGENDJAHRE'], axis=1)
    
    df_no_missing_values['CAMEO_INTL_2015_wealth'] = df_no_missing_values['CAMEO_INTL_2015'].apply(map_wealthy_dict)
    df_no_missing_values['CAMEO_INTL_2015_life_stage'] = df_no_missing_values['CAMEO_INTL_2015'].apply(map_life_stage_dict)
    df_no_missing_values = df_no_missing_values.drop(['CAMEO_INTL_2015'], axis=1)
    df_no_missing_values = df_no_missing_values.drop(['LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB','PLZ8_BAUMAX','MIN_GEBAEUDEJAHR'],
                 axis=1)

    # Return the cleaned dataframe.
    return df_no_missing_values


# In[4]:



azdias = pd.read_csv("Udacity_AZDIAS_Subset.csv",sep=";")
df = clean_data(azdias)


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[5]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputed_features = imputer.fit_transform(df)


# In[6]:


# Apply feature scaling to the general population demographics data.

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[df.columns].as_matrix())


# ### Discussion 2.1: Apply Feature Scaling
# 
# Imputing was attempted using "Most_frequent" strategy.After imputation, the features were standardized using standardScaler()

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[7]:


# Apply PCA to the data.
from sklearn.decomposition import PCA

pca = PCA()
pca_features = pca.fit_transform(scaled_features)


# In[23]:


# Investigate the variance accounted for by each principal component.
#This is an Udacity lecture code

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s" % ((str(vals[i]*100)[:3])), (ind[i], vals[i]), va="bottom", ha="center", fontsize=4.5)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
scree_plot(pca)


# In[32]:


#Find cumulative variance
pca1 = pca.explained_variance_ratio_.tolist()
print("Cumulative Variance 60 PCA :",round(np.sum(pca1[:60])*100))
print("Cumulative Variance 80 PCA :",round(np.sum(pca1[:80])*100))


# In[26]:


# Re-apply PCA to the data while selecting for number of components to retain.

pca = PCA(60)
pca_features = pca.fit_transform(scaled_features)


# In[32]:


scree_plot(pca)


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# From the scree plot , the cumulative explained variance is around 83% for the top 60 components. I feel this is a good cut off for PCA selection, as by dropping 40% of the dataset we only lose 17% of the variation in the data set.
# Hence I retained Top 60 components.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[39]:


#Calculate Explained Variance for each Component
dimensions = ['PC {}'.format(i) for i in range(1,len(pca.components_)+1)]
variance = pd.DataFrame(pca.explained_variance_ratio_,columns = ['EXPLAINED_VARIANCE']).round(4)
variance.index = dimensions
variance.head()


# In[41]:


#Calculate Weights of each component
weights = pd.DataFrame(pca.components_, columns=df.columns)
weights = weights.round(4)
weights.index = dimensions
weights.head()

#concat variance and weights into one dataframe
explain = pd.concat([variance, weights], axis = 1, sort=False, join_axes=[variance.index])
explain.head()


# In[42]:



#Print top 3 positive and negative weight for PC 5
print("PC 1 :")
print("---"*10)
print("Top Positive Weights :")
print("---"*10)
print(weights.iloc[1-1].sort_values(ascending=False)[:3])
print("---"*10)
print("Top Negative Weights :")
print("---"*10)
print(weights.iloc[1-1].sort_values(ascending=True)[:3])


# In[45]:


# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

#creating a function
def print_pc_values(pc_num,weight_num):
    weights = pd.DataFrame(pca.components_, columns=df.columns)
    weights = weights.round(4)
    weights.index = dimensions
    #weights.head()
    
    print("PC",pc_num," : ")
    print("---"*10)
    print("Top Positive Weights :")
    print("---"*10)
    print(weights.iloc[pc_num-1].sort_values(ascending=False)[:weight_num])
    print("---"*10)
    print("Top Negative Weights :")
    print("---"*10)
    print(weights.iloc[pc_num-1].sort_values(ascending=True)[:weight_num])
    
    


# In[46]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
print_pc_values(1,3)


# In[47]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print_pc_values(2,3)


# In[48]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print_pc_values(3,3)


# ### Discussion 2.3: Interpret Principal Components
# 
# Principal Component 1 appears to group variables related to household size, financial and social status
# 
# LP_STATUS_GROB_1.0  social status , low income earners
# <br>
# HH_EINKOMMEN_SCORE  Estimated household net income
# <br>
# PLZ8_ANTG3          Number of 6-10 family houses in the PLZ8 region
# <br>
# MOBI_REGIO          Movement Patterns
# <br>
# FINANZ_MINIMALIST   low financial interest
# <br>
# PLZ8_ANTG1          Number of 1-2 family houses in the PLZ8 region
# <br>
# ############################################################################
# <br>
# <br>
# 
# Principal Component 2 appears to be related to individuals age ,financial and energy consumption patterns
# 
# <br>
# ALTERSKATEGORIE_GROB    Estimated Age
# <br>
# FINANZ_VORSORGER        financial topology, be prepared
# <br>
# ZABEOTYP_3              energy consumption topology, fair supplied
# <br>
# Decade                  Decade of movement of person's youth
# <br>
# FINANZ_SPARER           financial topology,money-saver
# <br>
# SEMIO_REL               personality topology, religious
# <br>
# ############################################################################
# 
# <br>
# <br>
# Principal Component 3 is associate with individual's personality type and gender
# 
# SEMIO_VERT   Personality topology,dreamful
# <br>
# SEMIO_FAM    Personality topology,family-minded
# <br>
# SEMIO_SOZ    Personality topology,socially-minded
# <br>
# ANREDE_KZ    Gender
# <br>
# SEMIO_KAEM   Personality topology,combative attitude
# <br>
# SEMIO_DOM    Personality topology,dominant-minded
# <br>

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[9]:


# Over a number of different cluster counts...
from sklearn.cluster import KMeans

def calculate_Score(cluster_count):
    kmeans = KMeans(cluster_count)
    model_k = kmeans.fit(pca_features)
    
    return abs(model_k.score(pca_features))
    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.


# In[10]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

center = []
score = []
cluster_count = 20

for i in range(1,cluster_count):
    center.append(i)
    print(i," : ",center)
    score.append(calculate_Score(i))
    print(i," : ",score)

plt.plot(center, score, linestyle='-', marker='x', color='red')
plt.xlabel('centre')
plt.ylabel('score')
plt.xticks(np.arange(1, cluster_count, 1))
plt.grid()


# In[17]:


plt.plot(center, score, linestyle='-', marker='x', color='red')
plt.xlabel('centre')
plt.ylabel('score')
plt.ylim([40000000,70000000])
plt.xticks(np.arange(1, cluster_count, 1))
plt.grid()


# In[18]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
start_time = time.time()
print(start_time)
kmeans_pop = KMeans(15).fit(pca_features)

population_predict = kmeans_pop.predict(pca_features)

print("--- Run time: %s mins ---" % np.round(((time.time() - start_time)/60),2))


# ### Discussion 3.1: Apply Clustering to General Population
# 
# From the plot, no clear elbow is visible. However the scores seem to start stabilizing after 15 clusters. Hence the general population has been segmented into 15 clusters
# 

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[19]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')
print("Col:",customers.shape[0])
print("Row:",customers.shape[1])


# In[20]:


az = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')
print("Col:",az.shape[0])
print("Row:",az.shape[1])


# In[21]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

df = clean_data(customers)
display (df.head (n=5))


# In[22]:


df2 = clean_data(az)
display (df2.head (n=5))


# There appears to be a missing column. The next block is to find the missing column

# In[23]:


list1 =  list(df2.columns)
list2 = list(df.columns)

def diff(first_list, second_list):
    second_list = set(second_list)
    return [item for item in first_list if item not in second_list]

print(diff(list1, list2))
print(len(list1))
print(len(list2))


# Add the column GEBAEUDETYP_5.0
# 

# In[24]:



customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')

print("head",customers.head(2))

customers_add_row = customers.append(customers.xs(191651), ignore_index=True)
customers_add_row.tail()
customers_add_row.loc[191652,'GEBAEUDETYP'] = 5.0

customers_add_row['GEBAEUDETYP'].value_counts()
customers_add_row.tail(5)


# In[71]:


df = clean_data(customers_add_row)
display (df.head (n=5))


# In[27]:



imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputed_features = imputer.fit_transform(df)

scaler = StandardScaler()
standardized_customers = scaler.fit_transform(df[df.columns].as_matrix())

pca_customers = pca.transform(standardized_customers)


# In[28]:



# Predict 
customer_cluster = kmeans_pop.predict(pca_customers)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[29]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

customer_clust_df = pd.DataFrame (customer_cluster, columns = ['customer_clusters'])

customer_clust_df.hist ()

pop_clust_df = pd.DataFrame (population_predict, columns = ['_population_clusters'])

pop_clust_df.hist ()


# In[30]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
pop_clus = pd.Series(population_predict)
cust_clus = pd.Series(customer_cluster)


pop_cluster_prop = pd.Series(100*pop_clus.value_counts().sort_index()/len(pop_clus))
cust_cluster_prop = pd.Series(100*cust_clus.value_counts().sort_index()/len(cust_clus))
delta = ((cust_clus.value_counts().sort_index()/len(cust_clus))-(pop_clus.value_counts().sort_index()/len(pop_clus)))

aggr_df =  pd.concat([pop_cluster_prop, cust_cluster_prop,delta], axis=1).reset_index()
aggr_df.columns = ['Cluster','Population','Customer','Delta']
display(aggr_df)

aggr_df.plot(x="Cluster", y=["Population", "Customer"], kind="bar")
plt.grid()


# In[60]:


def plot_scaled_comparison(df_sample, kmeans, cluster):
    X = pd.DataFrame.from_dict(dict(zip(df_sample.columns,
    pca.inverse_transform(kmeans_pop.cluster_centers_[cluster]))), orient='index').rename(
    columns={0: 'feature_values'}).sort_values('feature_values', ascending=False)
    X['feature_values_abs'] = abs(X['feature_values'])
    pd.concat((X['feature_values'][:10], X['feature_values'][-10:]), axis=0).plot(kind='barh');


# In[61]:


#Customers are over reporesented in Cluster 3
plot_scaled_comparison(df, kmeans_pop, 3)


# In[62]:


#Customers are under represented in cluster 9
plot_scaled_comparison(df, kmeans_pop, 9)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# 
# In Cluster 3 , customers are over represented as compared to the populataion. This cluster has wealthy individuals such as investors (Finanz_anleger) or people belonging to wealthy households. Predominant characteristics of this cluster are Green_Avantgarde (Membership in environmental sustainability as part of youth) and HH_Einkommen_score (Household_income).
# 
# In Cluster 9, customer are under represented as compared to the population. This cluster is low income population characterised by FInanz_Minimalist(people with low financial interest) , SEMIO_RAT (rational population typology) ,LP_STATUS_FEIN (orientation seeking low income earners).
# 
# Clearly customers in cluster 3 are pivotal for the company while customers in cluster 9 aren't.

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




