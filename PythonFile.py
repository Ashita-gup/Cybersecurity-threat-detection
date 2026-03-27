#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name : Ashita Gupta
# Roll No. : 2022BCY0008
# Project for FUNDAMENTAL OF DATA SCIENCE
# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[2]:


# Loading the data
# Dataset source URL : https://www.kaggle.com/datasets/jancsg/cybersecurity-suspicious-web-threat-interactions
df = pd.read_csv('CloudWatch_Traffic_Web_Attack.csv')


# In[3]:


print("Columns and their data types:")
print(df.dtypes)


# In[4]:


# Data Preprocessing
# Converting time-related columns to datetime
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])


# In[5]:


# Handling missing values
print("Missing values before cleaning:")
print(df.isnull().sum())


# In[6]:


# Encoding categorical variables
le = LabelEncoder()
categorical_columns = ['src_ip', 'src_ip_country_code', 'protocol', 'dst_ip', 'rule_names', 'observation_name', 'source.meta', 'source.name', 'detection_types']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


# In[7]:


# Exploratory Data Analysis (EDA)
# Basic statistics
print(df.describe())


# In[8]:


# Box Plot
for col in ['bytes_in', 'bytes_out']:
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()


# In[9]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[10]:


# Feature Engineering
# Creating new features
df['duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['hour_of_day'] = df['creation_time'].dt.hour


# In[11]:


# Standardization
scaler = StandardScaler()
numerical_columns = ['bytes_in', 'bytes_out', 'duration', 'response.code', 'dst_port']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[12]:


# Anomaly Detection
# Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(df[['bytes_in', 'bytes_out', 'duration']])


# In[13]:


# Visualizing Anomalies
sns.scatterplot(data=df, x='bytes_in', y='bytes_out', hue='anomaly', palette='coolwarm')
plt.title('Anomaly Detection')
plt.show()


# In[14]:


# Classification
# Preparing data
X = df.drop(columns=['anomaly', 'creation_time', 'end_time', 'time'])
y = (df['anomaly'] == -1).astype(int)  # 1 for anomalies, 0 for normal


# In[15]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[16]:


# Defining models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}


# In[17]:


# Training and evaluating models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n{name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    if y_pred_prob is not None:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# In[19]:


# Cross-validation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} - Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")


# In[ ]:




