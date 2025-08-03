import sys
print("Python executable:", sys.executable)

import pandas as pd
import numpy as np
from collections import Counter as c
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import missingno as msno

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
data = pd.read_csv(r"C:\Users\Win-10\Downloads\Telegram Desktop\ckd-3\ckd\dataset\kidney_disease.csv")


# Manually assign column names
data.columns = ['id','age','blood_pressure','specific_gravity','albumin',
                'sugar','red_blood_cells','pus_cell','pus_cell_clumps','bacteria',
                'blood_glucose_random','blood_urea','serum_creatinine','sodium','potassium',
                'hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count',
                'hypertension','diabetesmellitus','coronary_artery_disease','appetite',
                'pedal_edema','anemia','class']

# Remove 'id' column as it's not useful for prediction
data.drop('id', axis=1, inplace=True)

# Fix string formatting issues (e.g., '\tno', ' yes')
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

print("\nFirst few rows:\n", data.head())
print("\nColumns:\n", data.columns.tolist())
print("\nDataset Info:")
data.info()

# Columns that look numeric but are stored as object
bad_numeric_cols = ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']

# Step 1: Convert bad numeric columns from strings to floats
for col in bad_numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Numeric columns (including converted ones)
numerical_cols = ['age', 'blood_glucose_random', 'blood_pressure', 'blood_urea', 'hemoglobin',
                  'packed_cell_volume', 'potassium', 'red_blood_cell_count',
                  'serum_creatinine', 'sodium', 'white_blood_cell_count']

# Categorical columns
categorical_cols = ['hypertension', 'pus_cell_clumps', 'appetite',
                    'albumin', 'pus_cell', 'red_blood_cells', 'coronary_artery_disease',
                    'bacteria', 'anemia', 'sugar', 'diabetesmellitus', 'pedal_edema',
                    'specific_gravity', 'class']

# Fill missing values in numerical columns with mean
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing values in categorical columns with mode
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

from collections import Counter as c

catcols = list(data.dtypes[data.dtypes == "O"].index.values)

for i in catcols:
    print('Columns :', i)
    print(c(data[i]))  # This prints the full Counter object
    print('*' * 120 + '\n')
    # Label Encoding of Categorical Columns

# 'specific_gravity', 'albumin', 'sugar' are numerical, so not included in catcols
catcols = ['anemia', 'pedal_edema', 'appetite', 'bacteria', 'class',
           'coronary_artery_disease', 'diabetesmellitus', 'hypertension',
           'pus_cell', 'pus_cell_clumps', 'red_blood_cells']  # Only text categorical columns

from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder

# Loop through each categorical column
for i in catcols:
    print("LABEL ENCODING OF:", i)
    LEi = LabelEncoder()  # Creating LabelEncoder object
    print(c(data[i]))  # Classes before transformation
    data[i] = LEi.fit_transform(data[i])  # Fit and transform to numerical values
    print(c(data[i]))  # Classes after transformation
    print("*" * 100)
# Step 1: Get columns with numeric (int/float) data types
contcols = set(data.dtypes[data.dtypes != 'O'].index.values)  # only fetch float and int columns
print(contcols)

# Step 2: Display value counts for all continuous columns
from collections import Counter as c

for i in contcols:
    print("Continuous Columns :", i)
    print(c(data[i]))
    print("*" * 120 + '\n')

# Step 3: Remove columns that have very few unique values (treated as categorical)
contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)

# Step 4: Add columns that were earlier converted to numeric but are actually continuous
contcols.add('red_blood_cell_count')  # using add() to add the column
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)
# Step 1: Add additional categorical columns that were originally removed
catcols.append('specific_gravity')
catcols.append('albumin')
catcols.append('sugar')
print(catcols)

# Step 2: Fix unwanted string entries like '\tno' or '\tyes' in specific columns

# Fix coronary_artery_disease values (e.g., '\tno' to 'no')
data['coronary_artery_disease'] = data['coronary_artery_disease'].replace('\tno', 'no')
print(c(data['coronary_artery_disease']))

# Fix diabetesmellitus values (e.g., '\tno' → 'no', '\tyes' → 'yes', ' yes' → 'yes')
data['diabetesmellitus'] = data['diabetesmellitus'].replace(
    to_replace={r'\tno': 'no', r'\tyes': 'yes', r' yes': 'yes'}, regex=True)
print(c(data['diabetesmellitus']))
print(data.describe())

# Scaling only the input (independent) variables
from sklearn.preprocessing import StandardScaler

# Selecting independent and dependent variables
selcols = ['red_blood_cells', 'pus_cell', 'blood_glucose_random', 'blood_urea',
           'pedal_edema', 'anemia', 'diabetesmellitus', 'coronary_artery_disease']

x = pd.DataFrame(data, columns=selcols)
y = pd.DataFrame(data, columns=['class'])

# Standardize x values
sc = StandardScaler()
x_bal = sc.fit_transform(x)

# (Optional) convert x_bal back to DataFrame with original column names
x_bal = pd.DataFrame(x_bal, columns=selcols)

# Print shapes to verify
print(x_bal.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_bal, y, test_size=0.2, random_state=2)
# Importing the Keras libraries and packages

       
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Creating ANN skeleton view
classification = Sequential()
classification.add(Dense(30, activation='relu'))
classification.add(Dense(128, activation='relu'))
classification.add(Dense(64, activation='relu'))
classification.add(Dense(32, activation='relu'))
classification.add(Dense(1, activation='sigmoid'))

# Compiling the ANN model
classification.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
classification.fit(x_train, y_train, batch_size=10, validation_split=0.2, epochs=100)
from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier with 10 trees and entropy as the criterion
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')

# Train the model
rfc.fit(x_train, y_train.values.ravel())



# Predict on test data
y_predict = rfc.predict(x_test)

# Predict on training data (for training accuracy, optional)
y_predict_train = rfc.predict(x_train)
from sklearn.tree import DecisionTreeClassifier
import numpy as np # Often needed for array handling

# Assuming x_train, y_train, x_test are already defined in your environment.
# For example, they might be loaded from a dataset or generated earlier in your script.

# Initialize the Decision Tree Classifier
# The 'criterion' parameter was cut off in the image; common values are 'gini' or 'entropy'.
# Assuming 'gini' as it's the default and frequently used.
dtc = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='gini') # Assumed 'criterion' value

# Train the Decision Tree Classifier
dtc.fit(x_train, y_train)

# The image also shows another initialization of DecisionTreeClassifier,
# which would overwrite the 'dtc' object if uncommented and executed:
# dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)
# If you intend to use this second configuration, you would need to fit it again:
# dtc.fit(x_train, y_train)

# Make predictions on the test set
y_predict = dtc.predict(x_test)
print(y_predict) # The image shows y_predict being printed/displayed

# The image shows an array output, which implies the previous print.
# array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]])


# Make predictions on the training set
y_predict_train = dtc.predict(x_train)
# logistic_regression_script.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Assuming X_train, y_train, X_test are defined elsewhere in your script or loaded prior to this.

# Initialize the Logistic Regression model
lgr = LogisticRegression()

# Fit the model to the training data
# Using .ravel() to reshape y_train as per the common warning message
lgr.fit(x_train, y_train)

# Make predictions on the test set
y_predict = lgr.predict(x_test)
# predict_with_trained_models.py

import numpy as np 
input_data = [[1, 1, 121.000000, 36.0, 0.0, 0.0, 1.0, 0]]

# --- Logistic Regression ---
print("# Logistic Regression")
y_pred_lgr = lgr.predict(input_data)
print(y_pred_lgr)

print(y_pred_lgr) # This will print it again, reflecting the image's output style


# --- Decisiontree classifier ---
print("\n# Decisiontree classifier")
y_pred_dtc = dtc.predict(input_data)
print(y_pred_dtc)
print(y_pred_dtc)


# --- Random Forest Classifier ---
print("\n# Random Forest Classifier")
y_pred_rfc = rfc.predict(input_data)
print(y_pred_rfc)
print(y_pred_rfc)
classification.save('ckd.h5') 
y_pred = classification.predict(x_test)
print(y_pred)
y_pred=(y_pred > 0.5)
print(y_pred)
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
def predict_exit(sample_value):
    """
    Predicts the likelihood of CKD based on the given sample features.

    Args:
        sample_value (list): A list of numerical features representing one sample.

    Returns:
        numpy.ndarray: The prediction output from the classifier (e.g., probability).
    """
    # Convert list to numpy array
    sample_value = np.array(sample_value)

    # Reshape because sample_value contains only 1 record
    sample_value = sample_value.reshape(1, -1)

    # Feature Scaling
    sample_value = sc.transform(sample_value)

    return classification.predict(sample_value)

# Example usage of the predict_exit function
# This input_data must match the number of features your models were trained on (e.g., 8 features for your CKD data)
input_data_for_test = [[1, 1, 121.000000, 36.0, 0.0, 0.0, 1.0, 0]]


test = predict_exit(input_data_for_test) # Call the function


if test[0][0] >= 0.5: # Threshold of 0.5 for binary classification
    print('Prediction: High chance of CKD!')
else:
    print('Prediction: Low chance of CKD.')
    from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Assuming x_train, y_train, x_test, y_test are already defined
# and loaded from your dataset.

dfs = []
models = [
    ('LogReg', LogisticRegression(max_iter=1000)), # Added max_iter to prevent convergence warning
    ('RF', RandomForestClassifier()),
    ('DecisionTree', DecisionTreeClassifier())
]

results = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['NO CKD', 'CKD']

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, x_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
    clf = model.fit(x_train, y_train.values.ravel())
    y_pred = clf.predict(x_test)
    print(name)
    print(classification_report(y_test, y_pred, target_names=target_names))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)

final = pd.concat(dfs, ignore_index=True)
print(final)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Often useful for array handling, though not strictly required by these lines.

# Assume y_test and y_predict are already defined.
# y_test: true labels for your test set
# y_predict: predictions made by your model on the test set

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for Logistic Regression model')
plt.show()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Generally good practice to include if working with numerical data/arrays

# Assume y_test and y_predict are already defined.
# y_test: true labels for your test set
# y_predict: predictions made by your model on the test set (specifically from RandomForestClassifier in this context)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for RandomForestClassifier')
plt.show()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Good practice for numerical operations

# Assume y_test and y_predict are already defined.
# y_test: true labels for your test set
# y_predict: predictions made by your model on the test set (specifically from DecisionTreeClassifier in this context)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for DecisionTreeClassifier')
plt.show()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for ANN model')
plt.show()
# Code from image_b3ac03.png
import pandas as pd
import numpy as np # Included for general good practice with numerical operations

# Assume 'final' DataFrame is already defined and structured as expected from
# the previous model comparison step where `pd.concat(dfs, ignore_index=True)` was used.
# It should contain a 'model' column with model names.



# Define time-related metrics
time_metrics = ['fit_time', 'score_time'] # fit time metrics

## PERFORMANCE METRICS
# Get DataFrame without fit data (i.e., performance metrics)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- Code from image_b2b7a3.png ---
dtc = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='gini')
dtc.fit(x_train, y_train)
y_predict_dtc_test = dtc.predict(x_test)
y_predict_dtc_train = dtc.predict(x_train)

# --- Code from image_b2bc5d.png ---
lgr = LogisticRegression()
lgr.fit(x_train, y_train)
y_predict_lgr_test = lgr.predict(x_test)

# --- Code from image_b2c3a3.png ---
# Assuming lgr, dtc, rfc are already trained models
input_data = [[1, 1, 121.000000, 36.0, 0.0, 0.0, 1.0, 0]]

y_pred_lgr_single = lgr.predict(input_data)
print(y_pred_lgr_single)
print(y_pred_lgr_single)

y_pred_dtc_single = dtc.predict(input_data)
print(y_pred_dtc_single)
print(y_pred_dtc_single)

# Assuming rfc is trained elsewhere, e.g., rfc.fit(x_train, y_train.ravel())
y_pred_rfc_single = rfc.predict(input_data)
print(y_pred_rfc_single)
print(y_pred_rfc_single)

# --- Code from image_b3b020.png ---
# Assuming results_long_nofit DataFrame is defined from previous steps
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Classification Metric')
plt.savefig('./benchmark_models_performance.png', dpi=300)
plt.show()
pickle.dump(lgr,open('CKD.pkl', 'wb'))  # Save the trained model to a file



from sklearn.metrics import accuracy_score

print("Logistic Regression Accuracy:", accuracy_score(y_test, lgr.predict(x_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dtc.predict(x_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rfc.predict(x_test)))
print("ANN Accuracy:", accuracy_score(y_test, (classification.predict(x_test) > 0.5)))
from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('CKD.pkl', 'rb'))  
@app.route('/')
def home():
    return render_template('home.html')   
@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def predict():
    # Reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array([input_features])

    features_name = [
        'blood_urea', 'blood glucose random', 'anemia',
        'coronary_artery_disease', 'pus_cell', 'red_blood_cells',
        'diabetesmellitus', 'pedal_edema'
    ]

    df = pd.DataFrame(features_value, columns=features_name)

    # Predictions using the loaded model file
    output = model.predict(df)
    return render_template('result.html', prediction_text=output)

    