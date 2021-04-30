#!/usr/bin/env python
# coding: utf-8

# # UNSW-NB 15 Dataset ML Methods Comparison Notebook - Attack or Normal Response
# 
# This notebook would implement several modified (Explainable AI (XAI)) machine learning methods:
# 
# - Decision Trees
# - Neural Networks: we will use Multi-layer perceptron.
# - XGBoost
# 
# We will illustrate them and compare their performances using the raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours. Further information found at https://www.kaggle.com/mrwellsdavid/unsw-nb15
# 
# 
# ![unsw-nb15-testbed.jpg](attachment:unsw-nb15-testbed.jpg)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# Tcpdump tool is utilised to capture 100 GB of the raw traffic (e.g., Pcap files). This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label.
# 
# These features are described in UNSW-NB15_features.csv file.
# 
# A partition from this dataset is configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively.
# 
# The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.Figure 1 and 2 show the testbed configuration dataset and the method of the feature creation of the UNSW-NB15, respectively.
# 
# The details of the UNSW-NB15 dataset are published in following the papers:
# 
# Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.
# Moustafa, Nour, and Jill Slay. "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset." Information Security Journal: A Global Perspective (2016): 1-14.
# Moustafa, Nour, et al. . "Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks." IEEE Transactions on Big Data (2017).
# Moustafa, Nour, et al. "Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models." Data Analytics and Decision Support for Cybersecurity. Springer, Cham, 2017. 127-156.
# 
# Free use of the UNSW-NB15 dataset for academic research purposes is hereby granted in perpetuity. Use for commercial purposes should be agreed by the authors. Nour Moustafa and Jill Slay have asserted their rights under the Copyright. To whom intend the use of the UNSW-NB15 dataset have to cite the above two papers.
# 
# 
# For more information, please contact the authors: Harshil Patel & Yuesheng Chen are a students in Industrial Engineering at Ohio State University, and they are interested in new Cyber threat intelligence approaches and the technology of Industry 4.0. 
# 
# In this notebook, the operations conducted include:
# 
# - Preprocessing the data to prepare for training XAI ML models.
# - Training XAI ML models based on cross-validation.
# - Evaluating XAI ML models based on testing data.

# # Libraries
# 
# Import libararies to implement the described machine learning methods using a few different `sklearn` algorithms.

# In[40]:


# data cleaning and plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn: data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# sklearn: train model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

# sklearn classifiers

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets, ensemble, model_selection
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier


# # Data Processing
# 
# ## Load Data
# 
# **UNSW-NB15: a comprehensive data set for network intrusion detection systems**
# 
# These features are described in UNSW-NB15_features.csv file.
# 
# A partition from this dataset is configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively.
# 
# The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.Figure 1 and 2 show the testbed configuration dataset and the method of the feature creation of the UNSW-NB15, respectively. The addtional features are as described in UNSW-NB15_features.csv file.
# 
# Response Variable:
# 
# attack_cat: This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms.
# 
# Label: 0 for normal and 1 for attack records
# 

# In[8]:


# Load data
initial_data = pd.read_csv('UNSW_NB15_training.csv')


# In[9]:


# Look at the first 5 rows
initial_data.head(n=5)


# In[10]:


# information of the data: 583 data points, 10 features' columns and 1 target column
initial_data.info()


# ## Tidy Data
# 
# ### Check missing values
# First, we should check if there are missing values in the dataset. We could see that four patients do not have the value of `Albumin_and_Globulin_Ratio`.

# In[11]:


# check if there are Null values
initial_data.isnull().sum()


# A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. Actually, there exists some strategies to impute missing values (see [here](https://scikit-learn.org/stable/modules/impute.html)). For simplicity, we will discard the four rows with missing values. 

# In[12]:


# Discard the rows with missing values
data_to_use = initial_data.dropna()

# Shape of the data: we could see that the number of rows remains the same as no null values were reported
data_to_use.shape


# ### Check imbalanced issue on y
# 
# First, we get the `X` and `y1` and `y2` .

# In[13]:


X = data_to_use.drop(axis=1, columns=['attack_cat']) # X is a dataframe
X = X.drop(axis=1, columns=['label'])


y1 = data_to_use['attack_cat'].values # y is an array
y2 = data_to_use['label'].values


# In[14]:


# Calculate Y2 ratio
def data_ratio(y2):
    '''
    Calculate Y2's ratio
    '''
    unique, count = np.unique(y2, return_counts=True)
    ratio = round(count[0]/count[1], 1)
    return f'{ratio}:1 ({count[0]}/{count[1]})'


# In[15]:


print('The class ratio for the original data:', data_ratio(y1))
plt.figure(figsize=(13,5))
sns.countplot(y1,label="Sum")
plt.show()

print('The class ratio for the original data:', data_ratio(y2))
sns.countplot(y2,label="Sum")
plt.show()


# We could see that the dataset is not perfectly balanced. There are some sampling techniques to deal with this issue. Here, we ignore this issue because we are aimed to implement several ML models to compare their performance. 

# ### Split training and testing data
# 
# It is important to split `X` and `y` as training set and testing set. Here, we will split the original data as 70% training set and 30% testing set. But the partition action from this dataset was pre configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively.
# 
# The number of records in the TESTING set is 175,341 records and the TRAINING set is 82,332 records from the different types, attack and normal. Figure above shows the testbed configuration dataset and the method of the feature creation of the UNSW-NB15, respectively.
# 
# Thus the follwing code will not be utilized

# In[18]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[19]:


#print('The class ratio in training data: ', data_ratio(y_train))
#print('The class ratio in testing data: ', data_ratio(y_test))


# In[20]:


# Load data
test_data = pd.read_csv('UNSW_NB15_testing.csv')
X_test = test_data.drop(axis=1, columns=['attack_cat']) # X_test is a dataframe
X_test = X_test.drop(axis=1, columns=['label'])


y1_test = test_data['attack_cat'].values # y is an array
y2_test = test_data['label'].values


# We will convert the orginal training data to the datframes called X_train, y1_train, y2_train

# In[21]:


X_train = X
y1_train = y1
y2_train = y2


# ### Transform training and testing data
# 
# #### Transformation on X_train, X_test
# 
# Scikit-learn provides a library of transformers. Like other estimators, these are represented by classes with a `fit` method, which learns model parameters (e.g. mean and standard deviation for normalization) from a training set, and a `transform` method which applies this transformation model to unseen data. 
# 
# **NOTE: The reason of performing transformation after splitting the original data is that we will `fit` those parameters on training set**.
# 
# In addition, it is very common to want to perform different data transformation techniques on different columns in your input data. The `ColumnTransformer` is a class in the scikit-learn library that allows you to selectively apply data preparation transforms. For example, it allows you to apply a specific transform or sequence of transforms to just the numerical columns, and a separate sequence of transforms to just the categorical columns.
# 
# In our case, we need to perform `OneHotEncoder` on `Gender` column because it is categorical, and perform `StandardScaler` on other numerical columns.
# 
# - `OneHotEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# - `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# 
# First, we find out which columns are categorical and which are numerical.

# In[22]:


# determine categorical and numerical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns


# In[23]:


numerical_cols


# In[24]:


categorical_cols


# Then, we construct the `ColumnTransformer` object, and then fit it on training data.

# In[25]:


# define the transformation methods for the columns
t = [('ohe', OneHotEncoder(drop='first'), categorical_cols),
    ('scale', StandardScaler(), numerical_cols)]

col_trans = ColumnTransformer(transformers=t)

# fit the transformation on training data
col_trans.fit(X_train)


# In[26]:


X_train_transform = col_trans.transform(X_train)


# In[27]:


# apply transformation to both training and testing data 
# fit the transformation on training data


# In[28]:


X_test_transform = col_trans.transform(X_test)


# We could look at the transformed training data. It becomes an array-like structure rather than a dataframe structure.

# In[30]:


# look at the transformed training data
X_train_transform.shape


# In[31]:


X_test_transform.shape


# #### Transformation on y_train and y_test
# 
# `LabelEncoder` is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1. Note that the distinct values/labels in `y` target are 1 ( no liver disease) and 2 (liver disease). In our case, we will transform the label 1 and 2 as 0 and 1, respectively. New label 0 refers to no disease and 1 refers to liver disease. Although it should be much easier to achieve this by subtracting 1 from the `y` array, we provide the `LabelEncoder` transformation which is a standard way of dealing with targeted values. Again, similar to the transformation on `X`, we will apply `fit` method to `y_train` and then apply `transform` method to both `y_train` and `y_test`.  

# In[32]:


# Note that the distinct values/labels in `y2` target are 1 and 2. 
pd.unique(y1)


# In[33]:


pd.unique(y2)


# In[34]:


# Define a LabelEncoder() transformation method and fit on y1_train
target_trans = LabelEncoder()
target_trans.fit(y1_train)


# In[35]:


# apply transformation method on y1_train and y1_test
y1_train_transform = target_trans.transform(y1_train)
y1_test_transform = target_trans.transform(y1_test)


# In[36]:


# view the transformed y1_train
y1_train_transform


# In[37]:


# Define a LabelEncoder() transformation method and fit on y2_train
target_trans = LabelEncoder()
target_trans.fit(y2_train)
y2_train_transform = target_trans.transform(y2_train)
y2_test_transform = target_trans.transform(y2_test)


# In[38]:


# view the transformed y2_train
y2_train_transform


# # Train Modified (Explainable AI (XAI)) ML Models
# 
# We will train several XAI machine learning models for the training set and evaluate their performance on both training and testing set. XAI VISULATION WILL BE CREATED ALSO
# 
# ## Steps of Training Model and Testing Model Performance with Testing Data
# 
# - Step 1: Train a XAI ML model and validate it via 5-fold cross-validation (CV). The CV results will show how good the model has been trained by using the training data given a set of hyperparameters in the ML model. The metrics of evaluating a model include accuracy, precision, recall, F1 score, AUC value of ROC. 
# 
# 
# - Step 2: Evaluate the model by using the testing data. It will show how good the model could be used to make predictions for unseen data.
# 
# **NOTE: For simplicity, we do not tune hyperparameters in the ML model and will use the default settings of hyperparameters in each ML model.**
# 
# **Let's firstly train a `Decision Tree Classifier` model with regards to Step 1 and Step 2.** using `y2` as the response feature FOR ALL THE XAI ML MODELS

# ## Explainable AI (XAI) with a Decision Tree Classifier

# In[134]:


# model and fit
DTclf = tree.DecisionTreeClassifier()
DTclf.fit(X_train_transform, y2_train_transform)


# In[135]:


feature_names = np.array(numerical_cols)
feature_names


# ### Feature importance: 
# 
# Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The most important features will be higher in the tree. A single feature can be used in different branches of the tree, feature importance then is its total contribution in reducing the impurity.

# In[74]:


importances = DTclf.feature_importances_
indices = np.argsort(importances)
features = np.array(numerical_cols)
plt.title('Feature Importances of Decision Tree Classifier')
count = 10 # top # importance
plt.barh(range(count), importances[indices][len(indices)-count:], color='g', align='center')
plt.yticks(range(count), [features[i] for i in indices[len(indices)-count:]])
plt.xlabel('Relative Importance')
plt.show()


# In[99]:


DTclf.feature_importances_


# #### In this case only the top 10 features are being used. The other features are not being used. Their importance is zero.

# In[133]:


get_ipython().system('pip install eli5')


# In[156]:


# Create Decision Tree classifer object
DTclf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
DTclf = DTclf.fit(X_train_transform, y2_train_transform)

#Predict the response for test dataset
y_pred = DTclf.predict(X_test_transform)


# In[157]:


from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y2_test_transform, y_pred))

report=metrics.classification_report(y2_test_transform,y_pred)

DTclf_name=['Decision Tree Classifer','RegLog']

print('Reporting for %s:'%DTclf_name)

print(report)


# In[163]:


fig = plt.figure(figsize=(35, 30))
DTtree = tree.plot_tree(DTclf, feature_names = np.array(numerical_cols), class_names = ['Normal', 'Attack'], fontsize=14, proportion=True, filled=True, rounded=True)
for o in DTtree:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('red')
        arrow.set_linewidth(3)

fig.savefig('Decision Tree Classifier XAI Visualization Part 2.png')


# In[164]:


import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(DTclf, random_state=1).fit(X_test_transform, y2_test_transform)
eli5.show_weights(perm, feature_names = np.array(numerical_cols))


# #### Let us visualize the first three levels of the decision tree, max_depth=3, 5, 8

# In[117]:


# visualization of DT Classfier
fig = plt.figure(figsize=(20, 16))
DTtree = tree.plot_tree(DTclf, feature_names = features, class_names = ['Normal', 'Attack'], fontsize=12, proportion=True, filled=True, rounded=True, max_depth=3) 
for o in DTtree:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('red')
        arrow.set_linewidth(3)

fig.savefig('Decision Tree Classifier (Depth = 3 Nodes) Explainable AI Visualization.png')


# In[118]:


# visualization of DT Classfier
fig = plt.figure(figsize=(20, 16))
DTtree = tree.plot_tree(DTclf, feature_names = features, class_names = ['Normal', 'Attack'], fontsize=12, proportion=True, filled=True, rounded=True, max_depth=5) 
for o in DTtree:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('red')
        arrow.set_linewidth(3)

fig.savefig('Decision Tree Classifier (Depth = 5 Nodes) Explainable AI Visualization.png')


# In[119]:


# visualization of DT Classfier
fig = plt.figure(figsize=(20, 16))
DTtree = tree.plot_tree(DTclf, feature_names = features, class_names = ['Normal', 'Attack'], fontsize=10, proportion=True, filled=True, rounded=True, max_depth=8) 
for o in DTtree:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('red')
        arrow.set_linewidth(3)

fig.savefig('Decision Tree Classifier (Depth = 8 Nodes) Explainable AI Visualization.png')


# ### Model Performance On Testing Set:

# In[122]:


# ===== Step 1: cross-validation ========
# define  Stratified 5-fold cross-validator, it provides train/validate indices to split data in train/validate sets.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

DTclf = tree.DecisionTreeClassifier()

# define metrics for evaluating
scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_ovr']  

# perform the 5-fold CV and get the metrics results
cv_results = cross_validate(estimator=DTclf,
                            X=X_train_transform,
                            y=y2_train_transform,
                            scoring=scoring,
                            cv=cv,
                            return_train_score=False) # prevent to show the train scores on cv splits.


# In[123]:


cv_results


# The CV results include: 
# 
# - `test_score`: The score array for test scores on each cv split. Suffix `_score` in `test_score` changes to a specific metric like `test_accuracy` or `test_f1` if there are multiple scoring metrics in the scoring parameter.
# 
# - `fit_time`: The time for fitting the estimator on the train set for each cv split.
# 
# - `score_time`: The time for scoring the estimator on the test set for each cv split. 
# 
# **Typically, we will use the mean value of each metric to represent the evaluation results of cross-validation.** For example, we could calculate the mean value of the `accuracy` score: 

# In[124]:


cv_results['test_accuracy'].mean()


# **In addition, the cross-validation step is used to find the best set of hyperparameters which give the "best" scores of metrics.** Since we do not tune hyperparameters in this case, we then directly fit the DT Classifer by using the default values of hyperparameters and evaluate it on testing data.

# In[125]:


# ======== Step 2: Evaluate the model using testing data =======

# fit the Logistic Regression model
DTclf.fit(X=X_train_transform, y=y2_train_transform)

# predition on testing data
y_pred_class = clf.predict(X=X_test_transform)
y_pred_score = clf.predict_proba(X=X_test_transform)[:, 1]

# AUC of ROC
auc_ontest = roc_auc_score(y_true=y2_test_transform, y_score=y_pred_score)
# confusion matrix
cm_ontest = confusion_matrix(y_true=y2_test_transform, y_pred=y_pred_class)
# precision score
precision_ontest = precision_score(y_true=y2_test_transform, y_pred=y_pred_class)
# recall score
recall_ontest = recall_score(y_true=y2_test_transform, y_pred=y_pred_class)
# classifition report
cls_report_ontest = classification_report(y_true=y2_test_transform, y_pred=y_pred_class)

# print the above results
print('The model scores {:1.5f} ROC AUC on the test set.'.format(auc_ontest))
print('The precision score on the test set: {:1.5f}'.format(precision_ontest))
print('The recall score on the test set: {:1.5f}'.format(recall_ontest))
print('Confusion Matrix:\n', cm_ontest)
# Print classification report:
print('Classification Report:\n', cls_report_ontest)


# Through the above steps, we could assess if the trained model is good for making predictions on unseen data. Recall that we are training a ML model to classify types of attack behavior on network packets. Considering that we prefer to having a model to capture the cases of attack as many as possible. In other words, the favorable model could have relatively high "coverage" ability and high "precision" ability. **Therefore, we could choose `F1 score` as the evaluation metric in this case. `F1 score` can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.** 

# ###### Exapand DT Classfier to RT Classfier to show XAI also

# ## Explainable AI (XAI) with a MLP (Multi-Layer Perceptron)

# Lime: Explaining the predictions of any machine learning classifier

# In[256]:


import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function
np.random.seed(1)


# In[258]:


MLP = MLPClassifier(random_state=123, solver='adam', max_iter=8000)
MLP.fit(X_train_transform, y2_train_transform)


# #### Explaining predictions

# In[259]:


metrics.accuracy_score(y2_test_transform, gbtree.predict(X_test_transform))


# In[260]:


predict_fn = lambda x: gbtree.predict_proba(x).astype(float)


# In[261]:


features = np.array(numerical_cols)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_transform ,feature_names = features, class_names = ['Normal', 'Attack'], kernel_width=3)


# In[265]:


np.random.seed(1)
i = 1653
exp = explainer.explain_instance(X_test_transform[i], predict_fn, num_features=10)
exp.show_in_notebook(show_all=False)
print('True class: ')
if (y2_test_transform[i] == 0):
    print('Normal')
else:
    print('Attack')


# ## Explainable AI (XAI) with XGBoost

# ###### XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. As it uses Gradient Boosting and can be parallelized, this algorithm is very popular in data science and is frequently used for regression and classification tasks. The following example shows a simple regression model and is hopefully a good entry point for anyone wanting to create and use XGBoost based models.

# ###### In an era of AI and ethics, explainability is one of the important recent topics in machine learning and data science. Let’s say you have built a machine learning model that performs well on your training and test data: how do you find out which samples and features offer the highest impact on your model’s output? This is where a library like SHAP can provide you with very valuable insights. SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explaining the output of any machine learning model. The following example shows how an XGBoost-based classifier model can be analyzed with SHAP to help better understand the impact of features on the model output. The chart on the top-right provides a view on the distribution of feature values and their impact on the model.

# ###### SHAP summary shows top feature contributions. It also shows data point distribution and provides visual indicators of how feature values affect predictions. Here red indicates higher feature value, blue indicates lower feature value. On the x-axis, higher SHAP value to the right corresponds to higher prediction value (more likely listing gets booked), lower SHAP value to the left corresponds to lower prediction value (less likely listing gets booked).

# #### Gradient boosting machine methods such as XGBoost are state-of-the-art for these types of prediction problems with tabular style input data of many modalities. Tree SHAP (arXiv paper) allows for the exact computation of SHAP values for tree ensemble methods, and has been integrated directly into the C++ XGBoost code base. This allows fast exact computation of SHAP values without sampling and without providing a background dataset (since the background is inferred from the coverage of the trees).
# 
# #### Here we demonstrate how to use SHAP values to understand XGBoost model predictions.

# In[167]:


get_ipython().system('pip install xgboost')


# In[168]:


get_ipython().system('pip install shap')


# In[169]:


from sklearn.model_selection import train_test_split
import xgboost
import shap
import numpy as np
import matplotlib.pylab as pl


# ##### Load Dataset

# In[171]:


xg_train = xgboost.DMatrix(X_train_transform, label=y2_train_transform)
xg_test = xgboost.DMatrix(X_test_transform, label=y2_test_transform)


# ##### Train Model

# In[173]:


params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y2_train_transform),
    "eval_metric": "logloss"
}
model = xgboost.train(params, xg_train, 5000, evals = [(xg_test, "test")], verbose_eval=100, early_stopping_rounds=20)


# ###### Explain predictions: Here we use the Tree SHAP implementation integrated into XGBoost to explain the testing dataset

# In[198]:


# this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
explainer = shap.TreeExplainer(model)
shap_values_test = explainer.shap_values(X_test)


# ##### Visualize a single prediction

# In[221]:


# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values_test[5,:], X_test.iloc[5,:])


# In[224]:


y2_test[5]


# ###### Visualize many predictions

# In[214]:


shap.force_plot(explainer.expected_value, shap_values_test[:1000,:], X_test.iloc[:1000,:])


# ###### Bar chart of mean importance

# In[212]:


shap.summary_plot(shap_values_test, X_test, plot_type="bar")


# SHAP Summary Plot:
# Rather than use a typical feature importance bar chart, we use a density scatter plot of SHAP values for each feature to identify how much impact each feature has on the model output for individuals in the validation dataset. Features are sorted by the sum of the SHAP value magnitudes across all samples. It is interesting to note that the relationship feature has more total model impact than the captial gain feature, but for those samples where capital gain matters it has more impact than age. In other words, capital gain effects a few predictions by a large amount, while age effects all predictions by a smaller amount.
# 
# Note that when the scatter points don't fit on a line they pile up to show density, and the color of each point represents the feature value of that individual.

# In[211]:


shap.summary_plot(shap_values_test, X_test)


# ###### SHAP Dependence Plots
# SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature's value vs. the SHAP value of that feature across many samples. SHAP dependence plots are similar to partial dependence plots, but account for the interaction effects present in the features, and are only defined in regions of the input space supported by data. The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, and another feature is chosen for coloring to highlight possible interactions.

# In[225]:


for name in X_test.columns:
    shap.dependence_plot(name, shap_values_test, X_test, display_features=X_test)


# ###### Compute SHAP Interaction Values
# 
# See the Tree SHAP paper for more details, but briefly, SHAP interaction values are a generalization of SHAP values to higher order interactions. Fast exact computation of pairwise interactions are implemented in the latest version of XGBoost with the pred_interactions flag. With this flag XGBoost returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions are divide them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model's current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect.

# In[227]:


# takes a couple minutes since SHAP interaction values take a factor of 2 * # features
# more time than SHAP values to compute, since this is just an example we only explain
# the first 2,000 people in order to run quicker
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test.iloc[:2000,:])


# ###### SHAP Interaction Value Summary Plot
# A summary plot of a SHAP interaction value matrix plots a matrix of summary plots with the main effects on the diagonal and the interaction effects off the diagonal.

# In[228]:


shap.summary_plot(shap_interaction_values, X_test.iloc[:2000,:])


# # USING LIME PACKAGE

# In[235]:


get_ipython().system('pip install lime')


# In[236]:


import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function
np.random.seed(1)


# In[245]:


import xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(X_train_transform, y2_train_transform)


# In[246]:


metrics.accuracy_score(y2_test_transform, gbtree.predict(X_test_transform))


# In[251]:


predict_fn = lambda x: gbtree.predict_proba(x).astype(float)


# #### Explaining predictions

# In[249]:


features = np.array(numerical_cols)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_transform ,feature_names = features, class_names = ['Normal', 'Attack'], kernel_width=3)


# These are just a mix of the continuous and categorical examples we showed before. For categorical features, the feature contribution is always the same as the linear model weight.

# In[252]:


np.random.seed(1)
i = 1653
exp = explainer.explain_instance(X_test_transform[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# In[254]:


i = 10
exp = explainer.explain_instance(X_test_transform[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# In[ ]:




