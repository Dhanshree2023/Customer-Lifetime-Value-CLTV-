#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the data into a pandas DataFrame
cltv_data=pd.read_csv(r"C:\Users\acer\Desktop\mini project 1.csv")


# In[3]:


cltv_data.head()


# In[4]:


#VAriable Identification
cltv_data.dtypes


# In[5]:


# Check for missing values
cltv_data.isnull().sum()


# # First Moment Business Decision 

# In[6]:


# Descriptive statistics
cltv_data.describe()


# In[7]:


cltv_data=cltv_data.drop(['id'],axis=1)


# In[8]:


# Calculate the counts of each gender
gender_counts = cltv_data['gender'].value_counts()

# Create a pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')

# Add title and legend
plt.title("Gender of customers")
plt.legend(title="Gender", labels=['Female', 'Male'])

# Show the plot
plt.show()


# # The dataset has more male customers than female customers.

# In[9]:


# Calculate the counts of each gender
gender_counts = cltv_data['area'].value_counts()

# Create a pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')

# Add title and legend
plt.title("area of customers")
plt.legend(title="area", labels=['Rural', 'Urban'])

# Show the plot
plt.show()


# # The customers are distributed among different areas, with the highest number in urban areas.

# In[10]:


def univariate_count(cltv_data,variable):
    #freq
    freq=cltv_data[variable].value_counts()
    #print summary statistics
    print('summary stat')
    print(freq)
    #plot bar plot
    cltv_data[variable].value_counts().plot(kind='bar')
    plt.xlabel(variable)
    plt.ylabel('frequency')
    plt.title('Type of policy')
    plt.show  


# In[11]:


univariate_count(cltv_data,'type_of_policy')


# # From above graph we can say that, maximum no. of customer prefer Platinum type of policy.

# In[12]:


# Bar chart of policy
sns.countplot(x='policy', data=cltv_data)
plt.title('Policy Distribution')
plt.show()


# From above graph we can say that, maximum no. of customer prefer Platinum type of policy and policy A.

# In[13]:


# Histogram of vintage
sns.histplot(cltv_data['vintage'])
plt.title('Vintage Distribution')
plt.show()

# Histogram of claim amount
sns.histplot(cltv_data['claim_amount'])
plt.title('Claim Amount Distribution')
plt.show()

# Histogram of number of policies
sns.histplot(cltv_data['marital_status'])
plt.title('Number of Policies Distribution')
plt.show()


# In above graph we can see that, 
# 1)most of the customers are associated with the insurance company for a long time.
# 2)Distribution if claim_amount is right-skewed.
# 3)The majority of customers are married.

# # Second Moment Business Decision(Bivariate Analysis)

# In[14]:


# Scatter plot of vintage and claim amount
sns.pairplot(cltv_data, x_vars=['vintage'], y_vars=['claim_amount'], height=5, aspect=1.2, kind='scatter')
plt.title('Vintage vs. Claim Amount')
plt.xlabel('Vintage')
plt.ylabel('Claim Amount')
plt.show()


# # The scatter plot of vintage and claim amount shows us that there is no clear relationship between the two variables.

# In[15]:



# Create a pair plot to visualize the relationships between income, vintage, and claim_amount
sns.pairplot(cltv_data[['income','marital_status','vintage', 'claim_amount','cltv']])
plt.show()


# # The scatter plot of vintage and claim amount shows us that there is relationship between claim_amount and cltv and no clear relationship between the other variables.

# In[16]:


# Box plot of claim amount by gender
sns.barplot(x='gender', y='claim_amount', data=cltv_data)
plt.title('Claim Amount by Gender')
plt.xlabel('Gender')
plt.ylabel('Claim Amount')
plt.show()


# # The bor plot of claim amount by gender shows us that males tend to have slightly higher claim amounts than females.

# In[17]:


#output variable distribution
plt.figure(figsize=(7,5))
sns.histplot(cltv_data['cltv'],bins=50,kde=True)
plt.title("distribution of customer lifetime value",size=10)
plt.show


# In[18]:


sns.boxplot(cltv_data['cltv'])


# # From the above histogram, we can see that the CLTV has a log tail distribution with outliers.

# In[19]:


# Identify the upper and lower limits for outliers in the cltv variable
Q1 = cltv_data['cltv'].quantile(0.25)
Q3 = cltv_data['cltv'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

# Remove outliers from the cltv variable
cltv = cltv_data[(cltv_data['cltv'] >= lower_limit) & (cltv_data['cltv'] <= upper_limit)]

# Create box plot of updated cltv variable
sns.boxplot(x=cltv['cltv'])


# # One Way ANOVA
# #compute the one way anova hypothesis test of correlation for categorical variable and output numeric variable.
# 
# H0(null hypothesis): The mean of group of each category of given feature variable is same.
# 
# H1(alternative hypothesis): There is difference between these two or more group means.

# In[20]:


# ANOVA TEST for - Gender
from scipy import stats

f_stats, p_value = stats.f_oneway(cltv_data[cltv_data['gender'] == 'Female']['cltv'],
                                  cltv_data[cltv_data['gender'] == 'Male']['cltv'])

print("ANOVA hypothesis test::")
print("F_statistics::", f_stats)
print("P-value::", p_value)

if p_value < 0.05:
    print(f"p-value={p_value:.3f}, Null hypothesis is rejected")
else:
    print(f"p-value={p_value:.3f} failed to reject null hypothesis.")


# # The above ANOVA test for Gender feature shows that there is statistical significant difference in means of customer lifetime value(cltv) of different gender cateogries of customers.

# In[21]:


# ANOVA TEST for - Area
f_stats, p_value = stats.f_oneway(cltv_data[cltv_data['area'] == 'Urban']['cltv'],
                                  cltv_data[cltv_data['area'] == 'Rural']['cltv'])

print("ANOVA hypothesis test::")
print("F_statistics::", f_stats)
print("P-value::", p_value)

if p_value < 0.05:
    print(f"p-value={p_value:.3f}, Null hypothesis is rejected")
else:
    print(f"p-value={p_value:.3f} failed to reject null hypothesis.")


# # The above ANOVA test for Area feature shows that there is statistical significant difference in means of customer lifetime value(cltv) of different Area cateogries of customers.

# In[22]:


# ANOVA TEST for - Qualification - 'Bachelor' 'High School' 'Others'
f_stats, p_value = stats.f_oneway(cltv_data[cltv_data['qualification'] == 'Bachelor']['cltv'],
                                  cltv_data[cltv_data['qualification'] == 'High School']['cltv'],
                                  cltv_data[cltv_data['qualification'] == 'Others']['cltv'])

print("ANOVA hypothesis test::")
print("F_statistics::", f_stats)
print("P-value::", p_value)
if p_value < 0.05:
    print(f"p-value={p_value:.3f}, Null hypothesis is rejected")
else:
    print(f"p-value={p_value:.3f}, failed to reject null hypothesis.")


# # The above ANOVA test for qualification feature shows that there is statistical significant difference between means of different education degrees.

# In[23]:


#ANOVA TEST for - Income - ['5L-10L', 'More than 10L', '<=2L', '2L-5L']
f_stats, p_value = stats.f_oneway(cltv_data[cltv_data['income'] == '5L-10L']['cltv'],
                                  cltv_data[cltv_data['income'] == 'More than 10L']['cltv'],
                                  cltv_data[cltv_data['income'] == '<=2L']['cltv'],
                                  cltv_data[cltv_data['income'] == '2L-5L']['cltv']
                                 )

print("ANOVA hypothesis test::")
print("F_statistics::", f_stats)
print("P-value::", p_value)

if p_value < 0.05:
    print(f"p-value={p_value:.3f}, Null hypothesis is rejected")
else:
    print(f"p-value={p_value:.3f}, failed to reject null hypothesis.")


# # The above ANOVA test for income feature shows that there is statistical significant difference in means of customer lifetime value(cltv) of different income cateogries of customers.

# In[24]:


#To check relationship between continuous variable.
correlation= cltv_data[['marital_status', 'claim_amount', 'vintage',"cltv"]].corr()['cltv']
correlation


# From above result we can say there is relationship between claim_amount and cltv.

# # Model Building

# In[25]:


# In dataset Most of the feature are object type by using map function orderly we convert these into integer.
Name=[]
for col in ['gender','area','qualification','income','num_policies','policy','type_of_policy', ]:
    names=cltv_data[col].unique().tolist()
    Name+=[names]
    print(names)


# In[26]:


gender_map=dict(zip(Name[0],[1,0]))
area_map=dict(zip(Name[1],[1,0]))
qualification_map=dict(zip(Name[2],[2,1,0]))
income_map=dict(zip(Name[3],[3,2,1,0]))
num_policies_map=dict(zip(Name[4],[1,0]))
policy_map=dict(zip(Name[5],[2,0,1]))
type_of_policy_map=dict(zip(Name[6],[2,1,0]))


# In[27]:


df=cltv_data.copy()


# In[28]:


df['gender']=df['gender'].map(gender_map)
df['area']=df['area'].map(area_map)
df['qualification']=df['qualification'].map(qualification_map)
df['income']=df['income'].map(income_map)
df['num_policies']=df['num_policies'].map(num_policies_map)
df['policy']=df['policy'].map(policy_map)
df['type_of_policy']=df['type_of_policy'].map(type_of_policy_map)


# In[29]:


df.head()


# In[30]:


df.dtypes


# In[31]:


# In above graphical representation we see that variable 'claim_amount' and 'cltv' are positively skewed .so we transform these variable using log transformation and sqrt transformation


# In[32]:


df['claim_amount']=np.sqrt(df['claim_amount'])


# In[33]:


df['cltv']=np.sqrt(df['cltv'])


# In[34]:


df.head()


# In[35]:


#Seperate features and Target variable 
X = df.drop(['cltv'],axis=1)
y = df['cltv']
X.shape,y.shape


# # Split the data into Train and Test Dataset

# In[36]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=59)


# # Gradient Boosting Algorithm 

# In[37]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_percentage_error


# In[38]:


gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# In[39]:


gbm.fit(train_X, train_y)


# In[40]:


# make predictions on test set
y_pred = gbm.predict(test_X)


# In[41]:


# compute test error (RMSE)
rmse = mean_absolute_percentage_error(test_y, y_pred)

print('GBM RMSE: {:.2f}'.format(rmse))


# In[42]:


# make predictions on train set
y_train_pred = gbm.predict(train_X)


# In[43]:


# compute train error (RMSE)
train_rmse = mean_absolute_percentage_error(train_y, y_train_pred)
print(f"Train RMSE: {train_rmse:.2f}")


# Here, in GBM algorithm training and testing error are nearly equal.. we can say best possible performance on the given data.

# In[ ]:




