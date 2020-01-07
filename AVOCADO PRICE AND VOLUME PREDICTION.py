
# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Visualizations
import scipy.stats as stats #Scientific and technical computing
import seaborn as sb #Graph visualizations


# In[2]:


# Reading csv using pandas
import pandas as pd 

adf = pd.read_csv('D:/Msc Data Analytics/SEMESTER 2/DATA MINING/ASSIGNMENT 2/avocado.csv')
print(adf.head())


# In[3]:


adf.info()


# In[4]:


adf.describe()


# In[5]:


#First column uncertain: No column name given, repeatin after 0-51 counts! strange!
adf.drop(['Unnamed: 0'],axis=1,inplace=True)
adf.head()


# In[6]:


#null check 
#adf.isnull() --> only gives binary mask 
adf.isnull().sum()


# In[8]:


adf_US = adf[adf['region']=='TotalUS']
adf_US_organic = adf_US[adf_US['type']=='organic']
adf_US_organic.head()


# In[9]:


plt.figure(figsize=(12,5))
plt.title("Distribution of Avg Price")
#average_price_fit = stats.norm.pdf(adf['AveragePrice'],np.mean(adf['AveragePrice']),np.std(adf['AveragePrice']))
plt.xlabel('Average Price')
plt.ylabel('Probability')
#plt.hist(adf['AveragePrice'],bins=40,color='g')
#plt.plot(adf['AveragePrice'],average_price_fit)
sb.distplot(adf_US_organic["AveragePrice"],hist=True,kde=True,rug=True,bins=100, color = 'b')


# In[10]:


#creating month column
adf['Date'] = pd.to_datetime(adf['Date'], format='%Y-%m-%d')
adf['Month']=adf['Date'].map(lambda x: x.month)
adf = adf.sort_values(by='Date')


# In[11]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="AveragePrice",hue='year',data=adf_US_organic,palette='magma')


# In[12]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Volume",hue='year',data=adf_US_organic,palette='magma',)


# In[13]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Bags",hue='year',data=adf_US_organic,palette='magma')
#sb.lineplot(x="Month", y="Total Volume",hue='year',data=adf_US_organic,palette='copper')


# In[14]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4046",hue='year',data=adf_US_organic,palette='magma',)


# In[15]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4225",hue='year',data=adf_US_organic,palette='magma')


# In[16]:


plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4770",hue='year',data=adf_US_organic,palette='magma')


# In[17]:


adf_US_organic = adf_US_organic.sort_values(by='Date')
# Valid = adf[(adf['year'] == 2017) | (adf['year'] == 2018)]
# Train = adf[(adf['year'] != 2017) & (adf['year'] != 2018)]
Train = adf_US_organic.sort_values(by='Date')


# In[19]:


from fbprophet import Prophet #works best with time series & robust to missing data 
import matplotlib.pyplot as plt


# In[20]:


m = Prophet()
date_volume = Train.rename(columns={'Date':'ds', 'Total Volume':'y'})
m.fit(date_volume)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[21]:


fig1 = m.plot(forecast)


# In[22]:


fig2 = m.plot_components(forecast)


# In[23]:


n = Prophet()
date_bags = Train.rename(columns={'Date':'ds', 'Total Bags':'y'})
n.fit(date_bags)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig2 = m.plot(forecast)


# In[24]:


fig2 = m.plot_components(forecast)

