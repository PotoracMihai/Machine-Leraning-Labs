#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycaret.regression import *

# Load the datasets
hour_data = pd.read_csv('hour.csv')
day_data = pd.read_csv('day.csv')


# In[2]:


# Preprocesarea datelor
# a) Vizualizarea primelor rânduri ale seturilor de date
print(hour_data.head())
print(day_data.head())


# In[3]:


pip install  streamlit


# In[4]:


# b) Vizualizarea statisticilor descriptive
print(hour_data.describe())
print(day_data.describe())


# In[5]:


# c) Tratarea valorilor lipsă
print("Missing values in hour_data:")
print(hour_data.isnull().sum())
print("Missing values in day_data:")
print(day_data.isnull().sum())


# In[6]:


# d) Detectarea și eliminarea outlierilor
# Definim o funcție pentru eliminarea outlierilor bazată pe IQR

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Aplicăm funcția pentru coloanele numerice
for col in ['casual', 'registered', 'cnt']:
    day_data = remove_outliers(day_data, col)
    hour_data = remove_outliers(hour_data, col)

# Agregarea datelor orare la nivel de zi
hourly_grouped = hour_data.groupby('dteday').agg({
    'casual': 'sum',
    'registered': 'sum'
}).reset_index()

# Îmbinarea setului de date zilnic cu datele agregate
merged_data = pd.merge(day_data, hourly_grouped, on='dteday', suffixes=('_daily', '_hourly'))

# Crearea etichetei 'rider_type'
merged_data['rider_type'] = merged_data.apply(
    lambda row: 'registered' if row['registered_hourly'] > row['casual_hourly'] else 'casual', axis=1
)

# Eliminarea coloanelor redundante
merged_data.drop(columns=['dteday', 'instant', 'casual_daily', 'registered_daily', 'casual_hourly', 'registered_hourly'], inplace=True)

# Împărțirea în seturi de antrenament și testare
train, test = train_test_split(merged_data, test_size=0.3, random_state=1, stratify=merged_data['rider_type'])

# Configurarea PyCaret pentru regresie
reg_setup = setup(data=train, target='cnt', remove_outliers=True, remove_multicollinearity=True)

# Compararea modelelor și alegerea celui mai bun
best_reg_model = compare_models()

# Crearea și finalizarea celui mai bun model
final_reg_model = finalize_model(best_reg_model)

# Realizarea predicțiilor pe setul de testare
predictions = predict_model(final_reg_model, data=test)
print(predictions.head())


# In[7]:


# Elaborarea mai multor modele de regresie și evaluarea lor
# a) Modelul de regresie liniară
linear_model = create_model('lr')
linear_predictions = predict_model(linear_model, data=test)
print("Linear Regression Predictions:")
print(linear_predictions.head())

# b) Modelul k-NN
knn_model = create_model('knn')
knn_predictions = predict_model(knn_model, data=test)
print("K-NN Regression Predictions:")
print(knn_predictions.head())

# Compararea modelelor și alegerea celui mai bun
best_reg_model = compare_models()

# Crearea și finalizarea celui mai bun model
final_reg_model = finalize_model(best_reg_model)

# Realizarea predicțiilor pe setul de testare
predictions = predict_model(final_reg_model, data=test)
print(predictions.head())


# In[8]:


# Compararea performanțelor modelelor
evaluate_models = pull()
print("Model Performance Comparison:")
print(evaluate_models)


# In[9]:


import streamlit as st

# Interfață Streamlit
st.title("Analiza modelelor de regresie")
st.subheader("Predicții pentru regresia liniară")
st.write(linear_predictions.head())

st.subheader("Predicții pentru k-NN")
st.write(knn_predictions.head())

st.subheader("Comparația modelelor")
st.write(evaluate_models)


# In[16]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[10]:


# Aggregate the hourly data by day
hourly_grouped = hour_data.groupby('dteday').agg({
    'casual': 'sum',
    'registered': 'sum'
}).reset_index()

# Merge the aggregated hourly data with the daily data
merged_data = pd.merge(day_data, hourly_grouped, on='dteday', suffixes=('_daily', '_hourly'))
df = merged_data
# Check the first few rows of the merged data to ensure it's correct
print(df.head())

print(df.info())

print(df.isnull().sum())


# In[11]:


# Imbalance Analysis Plots
import matplotlib.pyplot as plt

# Plot the imbalance in the day dataset
plt.figure(figsize=(10, 5))
plt.bar(["Casual", "Registered"], [day_data["casual"].sum(), day_data["registered"].sum()])
plt.title("Imbalance in Day Dataset (Total Riders)")
plt.ylabel("Total Count")
plt.show()

# Plot the imbalance in the hour dataset
plt.figure(figsize=(10, 5))
plt.bar(["Casual", "Registered"], [hour_data["casual"].sum(), hour_data["registered"].sum()])
plt.title("Imbalance in Hour Dataset (Total Riders)")
plt.ylabel("Total Count")
plt.show()

# Merge datasets on date (dteday) by summing up hourly data to daily level
merged_df = hour_data.groupby("dteday")[["casual", "registered"]].sum().reset_index()

# Plot the imbalance after merging
plt.figure(figsize=(10, 5))
plt.bar(["Casual", "Registered"], [merged_df["casual"].sum(), merged_df["registered"].sum()])
plt.title("Imbalance After Merging (Total Riders Per Day Summed)")
plt.ylabel("Total Count")
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split

# Compute rider_type before dropping columns
merged_data['rider_type'] = merged_data.apply(
    lambda row: 'registered' if row['registered_hourly'] > row['casual_hourly'] else 'casual', axis=1
)

# Drop any potential leakage columns
df = merged_data.drop(columns=['dteday', 'instant', 'casual_daily', 'registered_daily', 
                               'casual_hourly', 'registered_hourly'])

# Verify feature set before training
print(df.columns)

# Split into training and testing sets (80-20 split)
train, test = train_test_split(df, test_size=0.3, random_state=1, stratify=df['rider_type'])

# Display the shape of the split data
print(train.shape, test.shape)


# In[13]:


from pycaret.regression import *

# Setup PyCaret for model comparison with outlier removal disabled
reg_setup = setup(data=train, target='cnt', remove_outliers=False, remove_multicollinearity=True)

# Compare models and choose the best one
best_reg_model = compare_models()

# Create and finalize the best model
final_reg_model = finalize_model(best_reg_model)

# Make predictions on the test set
predictions = predict_model(final_reg_model, data=test)

# Display the results
print(predictions.head())

# Check dataset shape after setup to verify row count
print(get_config('X').shape)

test_predictions = predict_model(final_reg_model, data=test)
print(test_predictions.head())

get_config('X').columns



# In[14]:


# Creating models for the best estimators
lgbm = create_model('lightgbm')


# In[15]:


# Finaliszing model for predictions
model = finalize_model(lgbm)
predictions = predict_model(model, data = test)


# In[ ]:




