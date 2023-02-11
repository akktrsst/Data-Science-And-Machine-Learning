#!/usr/bin/env python
# coding: utf-8

# # Flight Price Prediction
# 
# Flight ticket prices can be something hard to guess, today we might see a price, check out the price of the same flight tomorrow, it will be a different story. We might have often heard travellers saying that flight ticket prices are so unpredictable. Huh! Here we take on the challenge! As data scientists, we are gonna prove that given the right data anything can be predicted. Here you will be provided with prices of flight tickets for various airlines between the months of March and June of 2019 and between various cities.     Size of training set: 10683 records Size of test set: 2671 records FEATURES: Airline: The name of the airline. Date_of_Journey: The date of the journey Source: The source from which the service begins. Destination: The destination where the service ends. Route: The route taken by the flight to reach the destination. Dep_Time: The time when the journey starts from the source. Arrival_Time: Time of arrival at the destination. Duration: Total duration of the flight. Total_Stops: Total stops between the source and destination. Additional_Info: Additional information about the flight Price: The price of the ticket

# In[1]:


import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df = pd.read_excel(r'F:/Data_Train.xlsx')
test_df = pd.read_excel(r'F:/Test_Set.xlsx')


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# In[5]:


big_df  = train_df.append(test_df,sort=False)


# In[6]:


big_df.head()


# In[7]:


big_df.tail()


# In[8]:


big_df.dtypes


# ## Feature Engineering

# In[9]:


big_df['Date'] = big_df['Date_of_Journey'].str.split('/').str[0]


# In[10]:


big_df['Month'] = big_df['Date_of_Journey'].str.split('/').str[1]
big_df['Year'] = big_df['Date_of_Journey'].str.split('/').str[2]


# In[11]:


big_df.head()


# In[12]:


big_df = big_df.drop(['Date_of_Journey'],axis=1)


# In[13]:


big_df['Date'] = big_df['Date'].astype(int)
big_df['Month'] = big_df['Month'].astype(int)
big_df['Year'] = big_df['Year'].astype(int)


# In[14]:


big_df.dtypes


# In[15]:


big_df['Arrival_Time'] = big_df['Arrival_Time'].str.split(' ').str[0]


# In[16]:


big_df.head()


# In[17]:


big_df[big_df['Total_Stops'].isnull()]


# In[18]:


big_df['Total_Stops'] = big_df['Total_Stops'].fillna('1 stop')


# In[19]:


big_df['Total_Stops'] = big_df['Total_Stops'].replace('non-stop','0 stop')


# In[20]:


big_df.head()


# In[21]:


big_df['Total_Stops'] = big_df['Total_Stops'].str.split(' ').str[0]


# In[22]:


big_df.head()


# In[23]:


big_df['Total_Stops'] = big_df['Total_Stops'].astype('int')


# In[24]:


big_df.dtypes


# In[25]:


big_df['Arrival_Hour'] = big_df['Arrival_Time'].str.split(':').str[0]
big_df['Arrival_Minute'] = big_df['Arrival_Time'].str.split(':').str[1]


# In[26]:


big_df['Arrival_Hour'] = big_df['Arrival_Hour'].astype('int')
big_df['Arrival_Minute'] = big_df['Arrival_Minute'].astype('int')


# In[27]:


big_df = big_df.drop(['Arrival_Time'],axis=1)


# In[28]:


big_df['Departure_Hour'] = big_df['Dep_Time'].str.split(':').str[0]
big_df['Departure_Minute'] = big_df['Dep_Time'].str.split(':').str[1]


# In[29]:


big_df['Departure_Hour'] = big_df['Departure_Hour'].astype('int')
big_df['Departure_Minute'] = big_df['Departure_Minute'].astype('int')


# In[30]:


big_df = big_df.drop(['Dep_Time'],axis=1)


# In[31]:


big_df.head()


# In[32]:


big_df['Duration_Hour'] = big_df['Duration'].str.split(' ').str[0]
big_df['Duration_Minute'] = big_df['Duration'].str.split(' ').str[1]


# In[33]:


big_df.head()


# In[34]:


big_df['Duration_Hour']= big_df['Duration_Hour'].str.split('h').str[0]
big_df['Duration_Minute'] = big_df['Duration_Minute'].str.split('m').str[0]


# In[35]:


big_df.head()


# In[36]:


big_df['Duration_Minute'] = big_df['Duration_Minute'].fillna(0)


# In[37]:


big_df['Duration_Hour'] = big_df['Duration_Hour'].fillna(0)


# In[38]:


big_df.head()


# In[39]:


big_df['Route_1'] = big_df['Route'].str.split('→ ').str[0]
big_df['Route_2'] = big_df['Route'].str.split('→ ').str[1]
big_df['Route_3'] = big_df['Route'].str.split('→ ').str[2]
big_df['Route_4'] = big_df['Route'].str.split('→ ').str[3]
big_df['Route_5'] = big_df['Route'].str.split('→ ').str[4]


# In[40]:


big_df.head()


# In[41]:


big_df['Route_1'].fillna('None',inplace=True)
big_df['Route_2'].fillna('None',inplace=True)
big_df['Route_3'].fillna('None',inplace=True)
big_df['Route_4'].fillna('None',inplace=True)
big_df['Route_5'].fillna('None',inplace=True)


# In[42]:


big_df.head()


# In[43]:


big_df.isnull().sum()


# In[44]:


sns.distplot(big_df['Price'])


# In[45]:


train_df.mean()


# In[46]:


big_df['Price'].mean(),big_df['Price'].max(),big_df['Price'].median(),big_df['Price'].mode()


# In[47]:


big_df['Price'].fillna(big_df['Price'].median(),inplace=True)


# In[48]:


big_df.isnull().sum()


# In[49]:


from sklearn.preprocessing import LabelEncoder


# In[50]:


encoder = LabelEncoder()
big_df['Airline'] = encoder.fit_transform(big_df['Airline'])
big_df['Source'] = encoder.fit_transform(big_df['Source'])
big_df['Destination'] = encoder.fit_transform(big_df['Destination'])
big_df['Route_1'] = encoder.fit_transform(big_df['Route_1'])
big_df['Route_2'] = encoder.fit_transform(big_df['Route_2'])
big_df['Route_3'] = encoder.fit_transform(big_df['Route_3'])
big_df['Route_4'] = encoder.fit_transform(big_df['Route_4'])
big_df['Route_5'] = encoder.fit_transform(big_df['Route_5'])


# In[51]:


big_df['Additional_Info'] = encoder.fit_transform(big_df['Additional_Info'])


# In[52]:


big_df.head()


# In[53]:


big_df.drop(['Route'],axis=1,inplace=True)


# In[54]:


big_df.drop(['Duration'],axis=1,inplace=True)


# In[55]:


big_df.head()


# In[56]:


big_df['Duration_Minute'] = big_df['Duration_Minute'].astype(int)


# In[57]:


big_df['Duration_Hour'] = big_df['Duration_Hour'].str.split('m').str[0]


# In[58]:


big_df['Duration_Hour'] = big_df['Duration_Hour'].astype(int)


# In[59]:


big_df.dtypes


# ## Feature Selection

# In[60]:


from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel


# In[61]:


df_train = big_df[0:10683]
df_test = big_df[10683:]


# In[62]:


X = df_train.drop(['Price'],axis=1)
y = df_train.Price


# In[63]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[64]:


model = SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[65]:


model.fit(X_train,y_train)


# In[66]:


model.get_support()


# In[67]:


selected_features = X_train.columns[(model.get_support())]


# In[68]:


selected_features


# In[69]:


X_train = X_train.drop(['Year'],axis=1)


# In[70]:


X_test = X_test.drop(['Year'],axis=1)


# ## Linear Regression

# In[71]:


from sklearn.linear_model import LinearRegression


# In[72]:


le_reg = LinearRegression()


# In[73]:


le_reg.fit(X_train,y_train)


# In[75]:


le_reg.intercept_


# In[76]:


le_reg.coef_


# In[77]:


print('The coefficient of determination R^2 for train set is: {}'.format(le_reg.score(X_train,y_train)))


# In[78]:


print('The coefficient of determination R^2 for test set is: {}'.format(le_reg.score(X_test,y_test)))


# In[81]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(le_reg,X,y,cv=5)


# In[83]:


score.mean()


# In[84]:


y_pred = le_reg.predict(X_test)


# In[85]:


sns.distplot(y_test-y_pred)


# In[87]:


plt.scatter(y_test,y_pred)


# In[90]:


from sklearn import metrics


# In[93]:


print('MAE: ', metrics.mean_absolute_error(y_test,y_pred))
print('MSE: ', metrics.mean_squared_error(y_test,y_pred))
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# ## Ridge Regression

# In[94]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ri_reg =Ridge()


# In[107]:


parameters = {'alpha':[0.4,0.6,0.004,0.003,0.0004,0.07,3,0.005]}
ri_model = GridSearchCV(estimator=ri_reg,param_grid=parameters,cv=5,scoring='neg_mean_squared_error')


# In[108]:


ri_model.fit(X_train,y_train)


# In[109]:


print(ri_model.best_params_)
print(ri_model.best_score_)


# In[116]:


print('MSE for train set is: {}'.format(ri_model.score(X_train,y_train)))
print('MSE for train set is: {}'.format(ri_model.score(X_test,y_test)))


# In[110]:


rid_pred = ri_model.predict(X_test)


# In[111]:


sns.distplot(y_test-rid_pred)


# In[114]:


plt.scatter(y_test,rid_pred)


# In[115]:


print('MAE: ', metrics.mean_absolute_error(y_test,rid_pred))
print('MSE: ', metrics.mean_squared_error(y_test,rid_pred))
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(y_test,rid_pred)))


# ## Lasso Regression

# In[117]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
la_reg = Lasso()


# In[118]:


para =  {'alpha' : [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}


# In[120]:


las_model = RandomizedSearchCV(la_reg,para,cv=5,scoring='neg_mean_squared_error')


# In[121]:


las_model.fit(X_train,y_train)


# In[123]:


las_model.best_params_ , las_model.best_score_


# In[126]:


las_pred = las_model.predict(X_test)


# In[127]:


sns.distplot(y_test-las_pred)


# In[129]:


plt.scatter(y_test,las_pred)


# In[130]:


print('MAE: ', metrics.mean_absolute_error(y_test,las_pred))
print('MSE: ', metrics.mean_squared_error(y_test,las_pred))
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(y_test,las_pred)))


# ## Decision Tree Regressor

# In[132]:


from sklearn.tree import DecisionTreeRegressor


# In[137]:


dt_reg = DecisionTreeRegressor()


# In[139]:


dt_reg.fit(X_train,y_train)


# In[140]:


dt_reg.score(X_train,y_train),dt_reg.score(X_test,y_test)


# In[141]:


score = cross_val_score(dt_reg,X_train,y_train,cv=5)


# In[142]:


score.mean()


# In[143]:


dt_pred = dt_reg.predict(X_test)


# In[144]:


sns.distplot(y_test-dt_pred)


# In[145]:


plt.scatter(y_test,dt_pred)


# In[159]:


print('MAE ', metrics.mean_absolute_error(y_test,dt_pred))
print('MSE ', metrics.mean_squared_error(y_test,dt_pred))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,dt_pred)))


# ### Hyperparameter optimization

# In[146]:


params = {"splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]}


# In[160]:


dtree_reg = GridSearchCV(dt_reg,params,cv=5,scoring='neg_mean_squared_error',n_jobs=-1,verbose=3)


# In[161]:


def time(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour,tsec = div_mod((datetime.now() - start_time).total_seconds(),3600)
        tmin,tsec = divmod(tsec,60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[162]:


from datetime import datetime
start_time = None
dtree_reg.fit(X_train,y_train)
time(start_time)


# In[163]:


dtree_reg.best_params_


# In[164]:


dtree_reg.best_score_


# In[165]:


dtree_pred =dtree_reg.predict(X_test)


# In[174]:


dtree_reg.score(X_train,y_train),dtree_reg.score(X_test,y_test)


# In[166]:


sns.distplot(y_test-dtree_pred)


# In[167]:


plt.scatter(y_test,dtree_pred)


# In[168]:


print('MAE ', metrics.mean_absolute_error(y_test,dtree_pred))
print('MSE ', metrics.mean_squared_error(y_test,dtree_pred))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,dtree_pred)))


# In[169]:


##conda install pydotplus
## conda install python-graphviz

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus


# In[171]:


features = list(X_train.columns)
features


# In[172]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# In[175]:


dot_data = StringIO()  
export_graphviz(dt_reg, out_file=dot_data,feature_names=features,filled=True,rounded=True,max_depth=5)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Random Forest

# In[176]:


from sklearn.ensemble import RandomForestRegressor


# In[177]:


rf_reg = RandomForestRegressor()


# In[178]:


rf_reg.fit(X_train,y_train)


# In[179]:


rf_reg.score(X_train,y_train),rf_reg.score(X_test,y_test)


# In[180]:


rf_pred = rf_reg.predict(X_test)


# In[181]:


sns.distplot(y_test-rf_pred)


# In[182]:


plt.scatter(y_test,rf_pred)


# In[183]:


print('MAE ', metrics.mean_absolute_error(y_test,rf_pred))
print('MSE ', metrics.mean_squared_error(y_test,rf_pred))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,rf_pred)))


# ### Hyperparameter tuning

# In[185]:


rf_reg.get_params


# In[186]:


n_estimators = [int(i) for i in np.linspace(100,1200,num=12)]

max_depth = [int(i) for i in np.linspace(5,30,num=6)]

min_weight_fraction_leaf =[0.1,0.2,0.4,0.5,0.7,0.6]

max_features = ['auto','sqrt']

min_samples_split = [4,5,7,10,15,100]

min_samples_leaf = [1,5,7,10]


# In[196]:


ran_params = {'n_estimators' : n_estimators,
             'max_depth' : max_depth,
             'max_features' : max_features,
             'min_samples_split': min_samples_split,
             'min_samples_leaf':min_samples_leaf}


# In[197]:


rand_reg = RandomizedSearchCV(rf_reg,param_distributions=ran_params,n_iter=100,cv=5,scoring='neg_mean_squared_error',verbose=3,n_jobs=-1)


# In[198]:


rand_reg.fit(X_train,y_train)


# In[200]:


rand_reg.best_params_


# In[203]:


rand_reg.best_score_


# In[204]:


rand_pred = rand_reg.predict(X_test)


# In[207]:


sns.distplot(y_test-rand_pred)


# In[209]:


plt.scatter(y_test,rand_pred)


# In[205]:


print('MAE ', metrics.mean_absolute_error(y_test,rand_pred))
print('MSE ', metrics.mean_squared_error(y_test,rand_pred))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,rand_pred)))


# In[212]:


import xgboost as xgb


# In[214]:


xg_reg = xgb.XGBRegressor()


# In[215]:


xg_reg.fit(X_train,y_train)


# In[216]:


xg_reg.score(X_train,y_train),xg_reg.score(X_test,y_test)


# In[217]:


xg_pred = xg_reg.predict(X_test)


# In[219]:


sns.distplot(y_test-xg_pred)


# In[220]:


plt.scatter(y_test,xg_pred)


# In[222]:


xg_reg.get_params()


# In[227]:


n_estimators = [int(i) for i in np.linspace(100,1200,num=12)]

learning_rate = [0.2,0.3,0.4,0.5,0.6]

max_depth = [int(i) for i in np.linspace(5,40,num=8)]

min_child_weight = [3,5,6,8,7]

subsample = [0.4,0.5,0.7,0.8,0.9]


# In[228]:


ran_params = {'n_estimators' : n_estimators,
             'max_depth' : max_depth,
             'learning_rate' : learning_rate,
             'min_child_weight': min_child_weight,
             'subsample':subsample}


# In[229]:


xg_r = RandomizedSearchCV(xg_reg,ran_params,n_iter=100,scoring='neg_mean_squared_error',cv=5,verbose=3)


# In[230]:


xg_r.fit(X_train,y_train)


# In[232]:


xg_r.score(X_train,y_train),xg_r.score(X_test,y_test)


# In[234]:


xg_r.best_params_ , xg_r.best_score_


# In[235]:


xg_pred = xg_r.predict(X_test)


# In[237]:


sns.distplot(y_test-xg_pred)


# In[238]:


plt.scatter(y_test,xg_pred)


# In[239]:


import pickle


# In[240]:


#saving model to disk

pickle.dump(rand_reg,open('model.pkl','wb'))


# In[ ]:


#Loading model to compare results
model = pickle.load(open('model.pkl','rb'))
print(model.predict)

