#!/usr/bin/env python
# coding: utf-8

# # Import necessary python library
#   

# In[1]:


import zipfile
import pandas as pd
import numpy as np
from zipfile import ZipFile
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # specifying the zip file name

# In[2]:


file_name = "Stress_dataset.zip"
  


# In[3]:


# Extract the files from the zipped folder
with zipfile.ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()


# extracting all the files
print('Extracting all the files now...')
print('Done!')


# In[4]:


files = glob.glob('C:/tithi/Term 2/combineeee_dataset/*.csv')

readFile_count = len(files)
print ("combineeee_dataset:", readFile_count);


# In[5]:


# Loop through each CSV file, read it as a DataFrame, and get its shape
for file in files:
    df = pd.read_csv(file)
    shape = df.shape
    print(f"File: {file}, Rows: {shape[0]}, Columns: {shape[1]}")
    missing_values = df.isnull().sum()
    print(f"File: {file}")
    print(missing_values)


# In[6]:


# Loop through each CSV file, read it as a DataFrame, and check data types
for file in files:
    df = pd.read_csv(file)
    data_types = df.dtypes
    print(f"File: {file}")
    print(data_types)


# In[7]:


import pandas as pd

# List of file paths
file_paths = [
    'C:/tithi/Term 2/combineeee_dataset\combined_acc.csv',
    'C:/tithi/Term 2/combineeee_dataset\combined_eda.csv',
    'C:/tithi/Term 2/combineeee_dataset\combined_hr.csv',
    'C:/tithi/Term 2/combineeee_dataset\combined_temp.csv'
]

# Load each CSV file into a DataFrame
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Merge DataFrames based on 'id' and 'datetime'
merged_df = dfs[0]  # Start with the first DataFrame
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on=['id', 'datetime'], how='outer')

# Specify the output path and save the merged DataFrame as CSV
output_path = 'C:/tithi/Term 2/merge_dataset.csv'
merged_df.to_csv(output_path, index=False)

# Display the first few rows of the merged DataFrame
print(merged_df.head())


# In[8]:


# Summary statistics
print(merged_df.describe())


# In[9]:


merged_df.shape


# In[10]:


# chceking the missing values
merged_df.isna().sum()


# In[11]:


# chceking the datatype
merged_df.dtypes


# In[12]:


# chceking the corelation
merged_df.corr()


# In[13]:


# loading the dataset
pd.set_option('display.max_columns', 100)
df = pd.read_csv('C:\\tithi\\Term 2\\Data science and decision making\\merged_data_labeled.csv')

df.head(5)


# In[14]:


# class variation
df['label'].value_counts()


# In[15]:


df['label'].value_counts(normalize = True)


# In[16]:


# Summary statistics
print(df.describe())


# In[17]:


# chceking the corelation
df.corr()


# In[18]:


# correlation of all columns with the class column
df.corr()['label']


# In[19]:


# imblacement of classes
class_normalized_counts = df['label'].value_counts(normalize = True)
class_normalized_counts


# In[20]:


# Expected equal distribution (assuming no bias)
expected_distribution = [0.5, 0.5]


# In[21]:


# Performing Chi-Square test for independence
chi2_stat, p_val = stats.chisquare(class_normalized_counts, expected_distribution)

print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_val)


# In[22]:


import matplotlib.pyplot as plt

# Create a bar plot to show class distribution
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Percentage')

# Make sure the tick labels match the classes present in your data
tick_labels = ['0', '2']

bars = plt.bar(tick_labels, class_normalized_counts, color='teal',
               align='center', alpha=0.7, edgecolor='black', linewidth=1)

# Adding percentage labels using autopct
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.1%}', ha='center', fontsize=10)

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()



# In[23]:


# finding strongly connected feature with the class column
class_corr = df.corr()['label']
class_corr = class_corr.sort_values(ascending = False)
class_corr[1:12]


# In[24]:


plt.title('Strongly Connected Feature with the Class column')
plt.xlabel('Correlation')
plt.ylabel('Feature')
plt.barh(class_corr.index[1:12],class_corr[1:12])
plt.show()


# In[25]:


plt.title('Feature Distribution')
sns.histplot(class_corr[1:12],bins = 20,kde = True,color = 'red')
plt.show()


# In[26]:


# drawing correlation heatmap combing 10 feature together
columns = [f'Feature_{i}' for i in range(1, 50)]
# Defining the number of features per heatmap
features_per_heatmap = 10

# Calculating the number of heatmaps needed
num_heatmaps = len(df.columns) // features_per_heatmap + 1

# Creating and displaying multiple heatmaps
for i in range(num_heatmaps):
    start_idx = i * features_per_heatmap
    end_idx = min(start_idx + features_per_heatmap, len(df.columns))
    subset_columns = df.columns[start_idx:end_idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[subset_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Heatmap - Features {start_idx+1} to {end_idx}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# In[27]:


X = df.drop(columns = ['label'])
y = df['label']


# In[28]:


# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Data preprocessing: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
lr = LogisticRegression(C=0.1, solver='liblinear')
lr.fit(X_train_scaled, y_train)

# Perform cross-validation on the training data
lr_scores = cross_val_score(lr, X_train, y_train, cv=5)

# Predict on the test set
y_pred = lr.predict(X_test_scaled)

# Predict on the validation set
y_validation_pred = lr.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the cross-validation scores
print("Cross-Validation Scores:", lr_scores)
print(f"Mean lr Score: {lr_scores.mean():.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[29]:


# Initialize the K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier()

# Perform cross-validation on the training data
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)  # You can adjust the number of folds

# Train the KNN model on the entire training set
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Predict on the validation set
y_validation_pred = knn.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the KNN model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy on Test Set: {accuracy:.2f}")

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print(f"Mean CV Score: {cv_scores.mean():.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[30]:


# Train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Perform cross-validation on the training data
gnb_scores = cross_val_score(knn, X_train, y_train, cv=5) 

# Predict on the test set
y_pred = gnb.predict(X_test)

# Predict on the validation set
y_validation_pred = gnb.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Print the cross-validation scores
print("Cross-Validation Scores:", gnb_scores)
print(f"Mean CV Score: {gnb_scores.mean():.2f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[31]:


# Train the Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_classifier.predict(X_test)

# Predict on the validation set
y_validation_pred = dt_classifier.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[32]:


# Train the C-Support Vector Classification (SVC) model
svc = SVC(C=1.0, kernel='linear', random_state=42)  # You can adjust C and kernel
svc.fit(X_train, y_train)

# Predict on the test set
y_pred = svc.predict(X_test)

# Predict on the validation set
y_validation_pred = svc.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[33]:


# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators

# Train the Random Forest model
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest.predict(X_test)

# Predict on the validation set
y_validation_pred = random_forest.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the Random Forest model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[34]:


# Initialize the base classifier (Decision Tree in this case)
base_classifier = DecisionTreeClassifier()

# Initialize the Bagging classifier
bagging = BaggingClassifier(base_classifier, n_estimators=100, random_state=42)  # You can adjust the number of estimators

# Train the Bagging model
bagging.fit(X_train, y_train)

# Predict on the test set
y_pred = bagging.predict(X_test)

# Predict on the validation set
y_validation_pred = bagging.predict(X_validation_scaled)

# Evaluate the model on validation set
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Evaluate the Bagging model
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[35]:


# base classifiers
lr = LogisticRegression(C=0.1,max_iter=1000, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(max_depth=23, random_state=0)

# create an AdaBoostClassifier with custom majority voting
class MajorityVotingEnsemble:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        predictions = [clf.predict(X) for clf in self.classifiers]
        majority_votes = []
        for i in range(len(X)):
            votes = [p[i] for p in predictions]
            majority_vote = max(set(votes), key=votes.count)
            majority_votes.append(majority_vote)
        return majority_votes

# creating a majority voting ensemble using AdaBoostClassifier
ensemble_clf = [lr, knn, dt]
ensemble = MajorityVotingEnsemble(ensemble_clf)


ensemble.fit(X_train, y_train)


# In[36]:


# Predict on the test set
y_pred_ensemble = ensemble.predict(X_test)

# Calculate accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {accuracy_ensemble:.2f}")

# Print the classification report for the ensemble
print("Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))


# In[37]:


# Data visualization
# Create histograms for numeric columns
numeric_columns = merged_df.select_dtypes(include=['float64']).columns
merged_df[numeric_columns].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()


# In[38]:


# Correlation matrix heatmap
correlation_matrix = merged_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[39]:


# Pair plots for numeric variables
sns.pairplot(merged_df[numeric_columns])
plt.show()


# In[40]:


# Save the DataFrame to the specified directory
save_directory = 'C:/tithi/Term 2/merged_data'
merged_df.to_csv(os.path.join(save_directory, 'merged_and_preprocessed_data.csv'), index=False)

print("Merged and preprocessed data saved to 'merged_and_preprocessed_data.csv'")


# # Extract the data  from the zipped folder

# In[41]:


data=pd.read_excel('SurveyResults.xlsx')


# # Print the first 5 rows of the dataset

# In[42]:


print('--- First 5 rows of the dataset ---')
print(data.head())


# # Explore the dataset by checking its shape, data types, and basic statistics:

# In[43]:


print('\n--- Shape of the dataset ---')
print(data.shape)


# In[44]:


print(data.dtypes)


# In[45]:


# print(data.describe())
print(data.describe(datetime_is_numeric=True))


# In[46]:


data.drop('Lack of supplies', axis=1, inplace=True)


# # check for missing values 

# # drop rows with missing values

# In[47]:


data.dropna(inplace=True)



# In[48]:


print(data)


# In[49]:


# There are no missing values, so here can move on to the next step


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[51]:


# Explore the distribution of the 'Stress level' column
print(data['Stress level'].describe())


# # convert 'Stress level' column to float

# In[52]:


# replace 'na' values with NaN
data['Stress level'] = pd.to_numeric(data['Stress level'], errors='coerce')


# In[53]:


data


# # Explore the correlation between different signals and the 'Stress level' column

# In[54]:


print(data.corr(numeric_only=True)['Stress level'])
# print(data.corr()['Stress level'], numeric_only=True)


# In[55]:


# compute correlation matrix
corr_matrix = data.corr(numeric_only=True)


# # Identify signals that might be better candidates for predicting stress

# In[56]:


correlations = data.corr(numeric_only=True)
plt.figure(figsize=(10,10))
plt.title("Correlation Matrix")
plt.imshow(correlations, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.show()


# # Split the dataset into training and testing sets (80% training, 20% testing)

# In[56]:


train = data.sample(frac=0.8, random_state=1)
test = data.drop(train.index)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




