#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# # Task 3 : Exploratory Data Analysis-Retail

# Objective :
# 
# ● Perform ‘Exploratory Data Analysis’ on dataset ‘SampleSuperstore’
# 
# ● As a business manager, try to find out the weak areas where you can work to make more profit.
# 
# ● What all business problems you can derive by exploring the data?

# # importing the libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Taking Dataset from csv_file
df = pd.read_csv("SampleSuperstore.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['State'].unique()


# In[6]:


print('Shape of data is:',df.shape)
print('Rows = ',df.shape[0])
print('Columns = ',df.shape[1])


# In[7]:


# Checking missing values
# As there all values are zero means we don't have any null values with every feature columns
df.isnull().sum().to_frame('Null_values')


# In[8]:


#This describes the statistics information with integer contains columns values only
df.describe()


# In[9]:


#Our dataset contains some float, int, string values datatypes.
df.info()


# In[10]:


# checking for duplicates
df.duplicated().sum()


# In[11]:


# Removing Duplicates
df = df.drop_duplicates()
df.head()


# In[12]:


# unique values count in each column
df.nunique()


# In[13]:


# Dropping irrelevants column that we have Postal Code and country
drop = df.drop(columns=['Postal Code','Country'], axis=1, inplace =True)


# In[14]:


df.head()


# In[15]:


df.shape


# In[16]:


# values in each column list
col_features = [features for features in df.columns]
for feature in col_features:
    print(feature,df[feature].unique())
    print("-"*75)


# In[17]:


df.corr()


# # Data Visualization

# In[18]:


plt.figure(figsize=(10,5))
df['City'].value_counts().plot(color ='red')
plt.title('Best Market Cities ')
plt.grid()
plt.show()


# In[15]:


# Let's check the category with subcategory visualization
plt.figure(figsize=(20,10))
plt.bar('Sub-Category','Category',data=df)
plt.title('Category vs Sub-Category')
plt.xlabel('Sub-Category')
plt.ylabel('Category')
plt.xticks(rotation=60)
plt.show()


# In[19]:


figsize=(15,20)
sns.pairplot(df,hue='Sub-Category')


# In[18]:


df.hist(bins=50,figsize=(20,15))
plt.show()


# In[17]:


# count the total repeated States
df['State'].value_counts()


# In[19]:


df['Category'].value_counts().plot()
plt.title('Categories on High')
plt.grid()
plt.show()


# In[26]:


print(df['Sub-Category'].value_counts())
plt.figure(figsize = (12,6))
sns.countplot(x=df['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[21]:


sns.countplot(x=df['Region'])


# In[20]:


df.groupby('State').sum().sort_values('Profit').tail()


# In[22]:


plt.figure(figsize=(8,6))
sns.scatterplot(df['Sales'],df['Profit'],color = 'red')
plt.title('Profit vs Sales')
plt.grid()
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()


# In[27]:


# The top 5 States generating most profits are California , New York , Washington , Michigan , Virginia


# In[32]:


# let's visualize the data from highest city Profit
newdf = df[df['State'].isin(['California','New York','Washington','Michigan','Virginia'])]


# In[33]:


# Top 5 states with Ship Mode
sns.countplot(x='State',hue='Ship Mode', data = newdf, palette ='rainbow')


# In[30]:


# the preferred shiping mode of the Top 5 States generating most profit was Standard Class


# In[31]:


sns.countplot(x='Segment' ,hue = 'Ship Mode', data =df,palette=['yellow','pink','red','teal'])


# In[32]:


# For consumer, Corporate, Home Office preferred shiping mode is Standard Class


# In[33]:


# profit vs shipmode
sns.barplot(x=df['Ship Mode'], y=df['Profit'],palette='rainbow')


# In[34]:


# the standard class had most orders, as well First class generated most profit


# # Now exploring weak areas Sales Profit

# In[34]:


# Sales profit with State
states =np.round(df.groupby('State').sum(), decimals=2).sort_values('Profit',ascending=False)
plt.figure(figsize=(10,6))
plt.title('Total State wise Profit/loss', fontsize=11)
sns.barplot(states.index,states.Profit)
plt.xticks(rotation=85)
plt.show()


# In[36]:


# Analysis on Sales loss
loss_sales = np.round(df[-(df.Profit) > 0], decimals=2).sort_values('Profit')


# In[37]:


#Sales loss in Region
region_loss = loss_sales.groupby('Region').sum()
sns.barplot(region_loss.index,region_loss.Profit)
plt.show()


# In[38]:


print("States with loss Region","\n","-"*105)
print(f"Central:",loss_sales[loss_sales.Region == "Central"].State.unique().tolist())
print(f"East:",loss_sales[loss_sales.Region == "East"].State.unique().tolist())
print(f"South:",loss_sales[loss_sales.Region == "South"].State.unique().tolist())
print(f"West:",loss_sales[loss_sales.Region == "West"].State.unique().tolist())


# In[23]:


# chairs and Furnishings
furn_state= df[(df['Sub-Category']=='chairs') | (df['Sub-Category']== "Furnishings")].sort_values('Profit')
plt.figure(figsize=(15,5))
plt.title('Profit - chairs & Furnishings',fontsize=15)
sns.barplot(furn_state['State'],furn_state.Profit)
plt.xticks(rotation=75)
plt.show()

Observation:- Texas and illinois are in loss with respect to Furnishing and chairs
# In[24]:


# Bookcases and Tables
Bc_Tb = df[(df['Sub-Category']=='Bookcases') | (df['Sub-Category']=='Tables')].sort_values('Profit')
plt.figure(figsize=(15,5))
plt.title('profit- Bookcases & Tables')
sns.barplot(Bc_Tb['State'],Bc_Tb.Profit)
plt.xticks(rotation=75)
plt.show()


# In[26]:


# Pairplot showing dependency of variables 
sns.pairplot(data=df.iloc[:,-3:], kind='scatter')
plt.show()


# In[53]:


sns.distplot(df['Discount'], color='red')
# most of the orders had no discount , followed by 0.2% discount


# In[79]:


# Analysis on Discount, Quantity & Profit
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].set_title('Discount vs Profit', fontsize=15)
ax[1].set_title('Quantity vs Profit', fontsize=15)
sns.lineplot(df.Discount, df.Profit, color='red', label='Profit Change', ax=ax[0])
sns.lineplot(df.Quantity, df.Profit, color='teal', label='Profit Change', ax=ax[1])
plt.show()


# In[83]:


# Plotting Profit Change with Discount
plt.figure(figsize=(10,5))
sns.scatterplot(df.Discount , df.Profit, hue = df.Profit, s=100)
plt.show()


# # Conclusion
1] Better minimize if supplying furniture i.e Bookcases,Tables and the items in other categories that result in loss
2] Texas & Illinois must drop the supply of furniture and items in Technology will enhance their profit (especially Copiers).
3] There must be no discount/low discount only then profit will raise.
4] Central region facing more loss in sales compared with others.
5] Texas & Illinois are the States where overall sales are in loss and particularly for furniture.
6] Supply of Furniture results in high loss - especially Tables & Bookcases
7] When discount increases, Sales Loss is increasing.
# # Thank You
