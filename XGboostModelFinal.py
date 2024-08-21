
# XGBoost model for classifying land use in historical maps 
#Based on new dataset with 12 features, 4 classes and X maps. 
# In this Python script we train an XG boost classifyer to read images of the Swedish Economic map (1:10 000) produced (1935-1978) and classify it into 4 classes;
# Graphics, Arable land, Forest and Water, and Open land. After classification Forest and Water was separated by masking water from current maps.

# #Install programs in the docker container with pip install
# #Python version 3.8
# #XGBoost 
# #gdal
# #geopandas
# #numpy
# #matplotlib
# #shapely
# #openpyxl
# #optuna
# #ipywidgets
# #shap
# #sklearn-pandas

# Import packages needed for this model

import pandas as pd
import geopandas as gpd
import numpy as np
from geopandas import GeoSeries, GeoDataFrame
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely import wkt
import glob
import os
import rasterio as rio
import rasterio.plot as rp
from rasterio.coords import BoundingBox
from shapely.geometry import box
from osgeo import gdal
from openpyxl import Workbook
import matplotlib.pyplot as plt



 

#### First we manually digitized training points which we classified into 4 classes in column 
#'Class' = (0=Graphics, 1=Arable, 2=ForestAndWater, 3=Open) and extracted attributes from Tiff-files into a table. 
#Forest and water are both in dark green color on the maps and are difficult to separate (Auffret 2017), but, 
#can easily be masked after the classification from current maps. We will use the same approach later on.

# This explains the different Features, all were calculated using Whitebox Tools
# 'FID' = The unique identifier for each point
# 'Class' = Our manually classified points (0=Graphics, 1=Arable, 2=ForestAndWater, 3=Open)
# 'Class_txt' = The 4 classes in text
# 'Red = Red band
# 'Green= Green band
# 'Blue = blue band
# 'Intensity = The intesity calculated from RGB
# 'Hue,= The hue calculated from RGB
# 'Saturation = The saturation calculated from RGB,  
# 'B_G = Blue band/Green band
# 'B_R= Blue band/Red band
# 'G_R = Green band/Red band
# 'GaussR10' = Gaussian filter calculated on Red band with a sigma of 10  
# 'MedianFR19= Median filter calculated on Red band with a kernel of 19 cells 
# 'MinFG3 = Minimum filter calculated on Green band with a kernel of 3 cells
# 'StDevFI11 = Standard Deviation filter calculated on Intensity with a kernel of 11 cells 
# 'StDevFR19 = Standard Deviation filter calculated on Red band with a kernel of 19 cells
# 'StDevFI49 =Standard Deviation filter calculated on Intensity with a kernel of 49 cells
# 'AverageR = The average of Red for the entire map
# 'AverageB = The average of Blue for the entire map 
# 'AverageG = The average of Green for the entire map
# 'AverageI = The average of Intensity for the entire map 
# 'AverageH = The average of Hue for the entire map  
# 'AverageS = The average of Saturation for the entire map 
# 'filename = The name of the map (stratify by this to evaluate on new maps and not mix traing/testing data from the same maps)
# 'Auffret_orig = The original classes from Auffret et al. (2017) 1=Arable land, 2=Open land (including urban areas), 3=Forest, 4= Water. map 'borders' have pixel value 0.
# 'Peatmap = The peat map from Ågren et al 2022 (a raster file in 2m resolution that was used to mask Water (water=0))
# 'Class5 = My classes, but i replaced 2=ForestAndWater --> 2=Forest and 4=Water
# 'Class5_txt = The 5 classes above in text
# 'Auffret_txt_orig = The original classes from Auffret et al. (2017) in text.
# 'Auffret_my_lables = Here I converted the original classes from Auffret et al. (2017) to my classes (1=Arable, 2=Forest, 3=Open, 4=Water))
# 'Join_Count' = identifyer of old and combined new dataset (just reminants from some old tests)
# Since we don't need all variables to train our model, drop the ones we don't need ('FID', 'Class_txt', 'Auffret_orig', 'Peatmap', 'Class5', 'Class5_txt', 'Auffret_txt_orig', 'Auffret_my_lables', 'Join_Count').

#Here we have successively reduced the number of features to see if we can get less complex model, 
#we have also lowered maximum depth, eta, col_sample_bytree, subsample 
# and incread min_child_wheight, lambda, alfa. 
# all this will make the model more conservative and less prone to overfitting.


#Columns after dropping: Index(['Class', 'Hue', 'Saturation',
#       'B_G', 'B_R', 'G_R', 'GaussR10', 'MedianFR19', 'MinFG3', 
#       'StDevFR19', 'StDevFI49', 'AverageR', 'AverageB', 'AverageG',
#       'AverageI', 'AverageH', 'AverageS', 'filename'],

# Pandas is used for data manipulation
import pandas as pd

# First investigate the file (DataFrame): Read in data and display first 5 rows
map_df = pd.DataFrame(pd.read_excel('/workspace/data/MapsForTraingTesting/XGBpoints/XGBpointsAlla/XGBpoints.xlsx')) 
# show the dataframe 
map_df.columns

# Print the original column names
print("Original columns:", map_df.columns)

# Make a copy of the DataFrame
xgb_df = map_df.copy()

# Print the list of columns in the DataFrame
print("Columns in XGBDataFrame:", xgb_df.columns)

# Specify columns to drop (here I drop columns we don't need to train the model) 
columns_to_drop = ['FID', 'Class_txt', 'Red', 'Green', 'Blue', 'Intensity', 'Saturation', 'StDevFI11', 'AverageB', 'AverageI', 'AverageR', 'Auffret_orig', 'Peatmap', 'Class5', 'Class5_txt', 'Auffret_txt_orig', 'Auffret_my_lables', 'Join_Count']

# Check for column existence
missing_columns = [col for col in columns_to_drop if col not in xgb_df.columns]
if missing_columns:
    print("Error: Columns not found in DataFrame:", missing_columns)
else:
    # Drop specified columns
    xgb_df.drop(columns=columns_to_drop, axis=1, inplace=True)
    print("Columns after dropping:", xgb_df.columns)

# Print the first 5 rows of the DataFrame
print(xgb_df.head())

# Count the number of observations for each class
class_counts = xgb_df['Class'].value_counts()

print(class_counts)

# Split the data into training data (80%) and testing data (20%). sklearn have a handy module for this purpose. 

from sklearn.model_selection import train_test_split 

# 'xgb_df' is our  DataFrame, the first column is the target variable ('y') and the rest are features ('x')

# Extract features (x), target variable (y), and stratification variable (stratify) from the DataFrame
y = xgb_df.iloc[:, 0]  # Assuming the first column is the target variable
x = xgb_df.iloc[:, 1:-1]  # Assuming the rest of the columns, except the last one, are features
stratify = xgb_df.iloc[:, -1]  # Assuming the last column is 'filename'

from sklearn.model_selection import GroupShuffleSplit

#stratify should be a list or array of the same length as x and y, where each element is the group identifier 
#of the corresponding sample (in our case, the map identifier). 
#I.e. splittion our our data into training and testing data, 
#but ensuring that all samples from the same map end up in the same subset.

# Create a GroupShuffleSplit object
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices for the train and test sets
train_idx, test_idx = next(gss.split(x, y, groups=stratify))

# Create the train and test sets
x_master_train, y_master_train = x.iloc[train_idx], y.iloc[train_idx]
x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

# Get the filenames of the maps in the test dataset
test_filenames = stratify.iloc[test_idx]

# Write the filenames to a file
with open('/workspace/data/XGBresults/test_filenames.txt', 'w') as f:
    for filename in test_filenames:
        f.write(f'{filename}\n')
#To remove duplicates from this list (as the dataset is sorted by class and not map)
# Read the file into a set
with open('/workspace/data/XGBresults/test_filenames.txt', 'r') as f:
    filenames = set(line.strip() for line in f)

# Sort the filenames
filenames = sorted(filenames)

# Count the unique filenames (count how many maps are in our test dataset)
count = len(filenames)
print(f'There are {count} unique maps in the test dataset.')

# Write the sorted, unique filenames back to the file
with open('/workspace/data/XGBresults/test_filenames.txt', 'w') as f:
    for filename in filenames:
        f.write(f'{filename}\n')


#Count how many maps are in our training dataset
# Get the indices of the training data
train_indices = x_master_train.index

# Get the filenames corresponding to these indices
train_filenames = stratify.loc[train_indices]

# Count the unique filenames
num_unique_maps = train_filenames.nunique()

print(f'There are {num_unique_maps} unique maps in the training dataset.')        

# Print the shapes of the resulting sets
print("x_master_train shape:", x_master_train.shape)
print("x_test shape:", x_test.shape)
print("y_master_train shape:", y_master_train.shape)
print("y_test shape:", y_test.shape)

# Now we can define and train (fit) the XGB model. First transform to dmatrix data structure for xgboost 

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report,cohen_kappa_score,f1_score,matthews_corrcoef
from sklearn.preprocessing import LabelEncoder 
import optuna
from optuna import Trial, visualization
import plotly
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt

#construct Dmatrics
d_master_train = xgb.DMatrix(x_master_train, label=y_master_train, enable_categorical=True)

# Test set
d_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

print("Unique labels in y_master_train:", np.unique(y_master_train))
print("Unique labels in y_test:", np.unique(y_test))


# # hyperparameter tuning of xgboost model with baysesian optimization using optuna framework
# XGBoost is designed to optimize both computational efficiency and model performance. 
#It incorporates parallelization and tree pruning techniques that can make it faster than traditional Random Forest implementations.
#Additionally, XGBoost utilizes a more optimized algorithm that reduces memory usage and computational overhead.

# Define the objective function for Bayesian optimization
def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 4,
        'tree_method': 'hist',
         # L2 regularization weight.
        'lambda': trial.suggest_float('lambda', 5, 20),
            # L1 regularization weight.
        'alpha': trial.suggest_float('alpha', 1, 5),
        #defines booster
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        # maximum depth of the tree, signifies complexity of the tree.
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'eta': trial.suggest_float('eta', 0.001, 0.10),
        # defines how selective algorithm is.
        'gamma': trial.suggest_float('gamma', 0, 2),
        # sampling according to each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.66),
        # sampling ratio for training data.
        'subsample': trial.suggest_float('subsample', 0.2, 0.45),
        # minimum child weight, larger the term more conservative the tree.
        'min_child_weight': trial.suggest_int('min_child_weight', 7, 10),
    }

    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=False)
        params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=False)

    # Perform stratified k-fold cross-validation on the XGBoost model
    scv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    score_mcc = []

    for fold, (train_idx, val_idx) in enumerate(scv.split(x_master_train, y_master_train)):
        x_train, y_train = x.iloc[train_idx, :], y[train_idx]
        x_val, y_val = x.iloc[val_idx, :], y[val_idx]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)

        # Print unique labels in y_val for each fold
        unique_labels = np.unique(y_val)
        print(f'Fold {fold + 1}, Unique labels in y_val: {unique_labels}')

        if not all(label in range(4) for label in unique_labels):
            raise ValueError(f"Invalid labels in fold {fold + 1}, expected labels in range [0, 4)")

        # Train the XGBoost model
        model = xgb.train(params, dtrain, num_boost_round=50, #set to 50 for final model
                          early_stopping_rounds=10, evals=[(dval, 'eval')], verbose_eval=True) #set to 10 for final model

        # Make predictions on the validation set
        y_val_pred = model.predict(dval)

        # Calculate the evaluation metrics
        mcc_score = matthews_corrcoef(y_val, y_val_pred)
        score_mcc.append(mcc_score)

        # Print the MCC for each fold
        print(f'Fold {fold + 1}, MCC: {mcc_score}')

    return np.mean(score_mcc)

# Perform Bayesian optimization
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=100, show_progress_bar=True) #I ran this 100 times to get the best parmas that I copied to the model training.
print('Best params: ', study.best_params)
print('Best score: ', study.best_value)


# Save best parameters to a text file
with open('/workspace/data/XGBresults/best_params.txt', 'w') as f:
    f.write('Best params: ' + str(study.best_params) + '\n')
    f.write('Best score: ' + str(study.best_value) + '\n')

#shap doesnt support dart
#study.trials_dataframe().sort_values('value', ascending = False)

# Save the results to a dataframe and a CSV file
df = study.trials_dataframe()
df.to_csv('/workspace/data/XGBresults/optuna_map_results_nostone_Model14.csv', index=False)

import plotly.io as pio

fig = optuna.visualization.plot_optimization_history(study)
pio.write_image(fig, '/workspace/data/XGBresults/optimization_history.png')

fig = optuna.visualization.plot_param_importances(study)
pio.write_image(fig, '/workspace/data/XGBresults/param_importances.png')

fig = optuna.visualization.plot_slice(study, params=["gamma", "eta", "colsample_bytree", "lambda"])
pio.write_image(fig, '/workspace/data/XGBresults/slice_plot1.png')

fig = optuna.visualization.plot_slice(study, params=["alpha", "subsample", "max_depth", "min_child_weight"])
pio.write_image(fig, '/workspace/data/XGBresults/slice_plot2.png')

plt.close('all')  # Close all figures

# model training


# declare parameters (change to the best ones from above)
best_params = study.best_params
best_params['num_class'] = 4
best_params['verbosity'] = 1
best_params['eval_metric'] = ["mlogloss","merror"]

#training the model
evals = [(d_test, 'testing'), (d_master_train, 'training')]
evals_result = {}
# Assign name to my XGBoost model
xgboost_model2 = xgb.train(best_params, d_master_train, num_boost_round=100, evals=evals, early_stopping_rounds=10, evals_result=evals_result, verbose_eval=True)



# Print model performance
# Assuming evals_result is a dictionary with keys 'training' and 'testing' containing 'mlogloss'
plt.plot(range(len(evals_result['testing']['mlogloss'])), evals_result['testing']['mlogloss'], label="testing")
plt.plot(range(len(evals_result['training']['mlogloss'])), evals_result['training']['mlogloss'], label="training")
plt.legend()
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Training Process')

# Save the figure to a file
plt.savefig('/workspace/data/XGBresults/xgboost_losscurve.png', dpi=300)
plt.show()


test_pred = xgboost_model2.predict(d_test)
test_accuracy = accuracy_score(y_test, test_pred)
print("testing accuracy:", test_accuracy)
print(classification_report(y_test, test_pred))

print('cohens kappa score for testing:', cohen_kappa_score(y_test, test_pred))
print('model f1 score for testing:', f1_score(y_test, test_pred, average='macro'))
print('individual class f1 score for testing:', f1_score(y_test, test_pred, average=None))
print('mcc score for testing:', matthews_corrcoef(y_test,test_pred))

print('Remember that the land use classes were: 0=Graphics, 1=Arable, 2=ForestandWater, 3=Open')


from xgboost import plot_importance
#plot feature importance to file
plt.rcParams["figure.figsize"] = (10, 6)

plot_importance(xgboost_model2)
plt.title('xgboost feature importance')
plt.savefig('/workspace/data/XGBresults/VIP.png', dpi=300)
plt.show()

# SHAP values are a way to estimate the contribution of each feature to the model's prediction.
# SHAP values are calculated by averaging the loss function over all possible permutations of the input features.

# Calculate SHAP values
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import shap
explainer = shap.TreeExplainer(xgboost_model2)
shap_values = explainer.shap_values(x_test)
# Create a new figure
plt.figure()
shap.summary_plot(shap_values, x_test, plot_type="bar")
num_outputs = len(shap_values)

# class names
classes = ['Graphics', 'Arable', 'Foreast and Water', 'Open land']

# set RGB tuple per class
colors = [(1/255, 0, 40/255), (254/255, 255/255, 1/255), (39/255, 115/255, 0), (231/255, 254/255, 200/255)]

# get class ordering from shap values
class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])

# create listed colormap
cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

# plot
shap.summary_plot(shap_values, x_test, feature_names=x_test.columns, color=cmap, class_names=classes)

plt.savefig('/workspace/data/XGBresults/summary_shap_plot_newsummarybar.png')
# Close the figure to prevent it from being displayed
plt.close()

#Code from earlier models which calculates the correlation between the features in the dataset.
# This was useful for identifying redundant features and for understanding the relationships between the features.
#Fetures that covary a lot dont't add new infromation to the model, and were removed. 

import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix
corr = xgb_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Save the figure
plt.savefig('/workspace/data/XGBresults/correlation_heatmap.png')

plt.show()



# Save the model to a file that can be imported in the next step for predictions.
import pickle
# Specify the desired file path
model_file_path = '/workspace/data/XGBresults/xgboost_model20.pkl'
# Save the model to the specified file path
with open(model_file_path, 'wb') as f:
    pickle.dump(xgboost_model2, f)

print('Done')