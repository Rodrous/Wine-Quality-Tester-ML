
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve



data = pd.read_csv("WineTest.csv")


data.head(5)


data.quality.value_counts()

#DATA EXPLORATION

data.quality.value_counts()
data.quality.value_counts(normalize=True)*100
data.quality.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
data.describe()

#Relation between Alcohol and Quality of wine

pd.crosstab(data.quality, data.alcohol).plot(kind="bar",
                                    figsize=(10,6),
                                    color=["salmon", "lightblue"],legend = False);

plt.ylabel("Alcohol")

#Correlation between independent variable

corr_matrix = data.corr()
print(corr_matrix)
plt.figure(figsize=(25, 20))
sns.heatmap(corr_matrix,
            annot=True,
            linewidths=0.5,
            fmt= ".2f",
            cmap="gist_rainbow");


#PreProcessing Datas


bins = (2, 6.5, 8)
group_names = [0, 1]
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


data.isna().sum()
data.dropna(how = 'any',inplace = True)


data.quality.unique()

label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])


#Modeling
data.head()
# Everything except target variable
x = data.drop("quality", axis=1)

# Target variable
y = data.quality.values

#Splitting the data into Training and testing


# Random seed for reproducibility
np.random.seed(42)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(x, # independent variables
                                                    y, # dependent variable
                                                    test_size = 0.2) # percentage of data to use for test set


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(),
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores



model_scores = fit_and_score(models,X_train,X_test,y_train,y_test)
print(model_scores)



model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();


#hypertuning the model
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=10,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);
print(rs_rf.best_params_)

rs_rf.score(X_test,y_test)

#Evaluating the model beyond accuracy


# Make preidctions on test data
y_preds = rs_rf.predict(X_test)


#ROC Curves
# Plot ROC curve and calculate AUC metric
plot_roc_curve(rs_rf, X_test, y_test);

#Confusion Matrix


print(confusion_matrix(y_test, y_preds))

sns.set(font_scale=1.5)  # Increase font size


def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,  # Annotate the boxes
                     cbar=False,
                     cmap="plasma")
    plt.xlabel("true label")
    plt.ylabel("predicted label")


plot_conf_mat(y_test, y_preds)

#Classification report

print(classification_report(y_test, y_preds))



print(rs_rf.best_params_)




clf = RandomForestClassifier(n_estimators=510, min_samples_split=14,min_samples_leaf=1,max_depth=None)

#Cross Val Score
cross_val = cross_val_score(clf,
                         x,
                         y,
                         cv=10, # 10-fold cross-validation
                         scoring="accuracy") # accuracy as scoring
print(cross_val)

#Accuracy of the matrix

cv_acc = np.mean(cross_val)
print(f"Accuracy:- {cv_acc * 100}%")

#Precision
cv_precision = np.mean(cross_val_score(clf,
                                       x,
                                       y,
                                       cv=10, # 10-fold cross-validation
                                       scoring="precision")) # precision as scoring
print(f"Precision:- {cv_precision*100}%")

#Cross-validated recall score
cv_recall = np.mean(cross_val_score(clf,
                                    x,
                                    y,
                                    cv=10, # 5-fold cross-validation
                                    scoring="recall")) # recall as scoring
print(f"Cross-val recall score:- {cv_recall*100}%")

#Cross-validated F1 score
cv_f1 = np.mean(cross_val_score(clf,
                                x,
                                y,
                                cv=10, # 5-fold cross-validation
                                scoring="f1")) # f1 as scoring
print(f"Cross-val f1 score:- {cv_f1*100}%")

#Visualizing this data
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                           "F1": cv_f1
                            },
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);

#Feature importance



clf.fit(X_train, y_train);

features_dict = dict(zip(data.columns, list(clf.feature_importances_)))
features_df = pd.DataFrame(features_dict, index = [1])

features_df.T.plot.bar(title="Feature Importance", legend = False)


#Exporting the Model
from joblib import dump, load
dump(clf, 'Winequality.joblib')

#Extras


def predictions(model,dataset):
    print(model.predict(dataset))

