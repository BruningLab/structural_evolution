"""
XGBoost model training for selection of features for Markov modelling. 

Author: Daniel P. McDougal

This code takes as input a csv file containing variance threshold filtered features and a label column
assigning each receptor e.g., "hESR1", "rfESR1" etc
      
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



# load the dataset as a pandas dataframe and Z-score normalise features 
df = pd.read_csv("ER_LBD_filtered_features.csv", index_col=[0]) #pre-filtered with a variance threshold of 0.1
feature_columns = [c for c in df.columns if c not in ["label"]]

X = np.array(df.loc[:, feature_columns])
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std



# encode class labels as integers and partition the data into train/test splits (70:30)
label_encoder = LabelEncoder()
y = np.array(label_encoder.fit_transform(df["label"]))
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42, shuffle=True)



# initiate the XGBoost model and tune hyperparameters with RandomizedSearchCV
model = XGBClassifier(verbosity=0)

param_dict = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 150, 200],
    "subsample": [0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0]}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dict, 
                                   n_iter=10, cv=5, scoring="precision_weighted", 
                                   refit=True, verbose=3, random_state=42)

random_search.fit(X_train, y_train) # this could take some time... have a coffee break



# train the model and fit with the best hyperparameters from RandomisedSearchCV
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = random_search.best_estimator_
best_model.fit(X_train, y_train, early_stopping_rounds=5, 
               eval_set=[(X_train, y_train), (X_test, y_test)], 
               eval_metric=["aucpr", "mlogloss", "merror"], verbose=True)



# save the model
joblib.dump(best_model, "xgboost_best_model.joblib")



# evaluate performance of the model
y_pred = best_model.predict(X_test) #make predictions

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Best Hyperparameters: %.2f%%" % (accuracy * 100))
print("Precision: %.4f" % precision_score(y_test, y_pred, average="weighted"))
print("Recall: %.4f" % recall_score(y_test, y_pred, average="weighted"))
print("F1 Score: %.4f" % f1_score(y_test, y_pred, average="weighted"))



cm = confusion_matrix(y_test, y_pred)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
print(cm)

confusion_matrix = pd.DataFrame(data=cm, 
                                index=["$h$ER\u03B1", "$h$ER\u03B2", "$rf$ER\u03B1", "$rf$ER\u03B2", "$rf$ER\u03B3"],
                                columns=["$h$ER\u03B1", "$h$ER\u03B2", "$rf$ER\u03B1", "$rf$ER\u03B2", "$rf$ER\u03B3"])

sns.set(rc={'figure.figsize': (3.5, 2.5)})
sns.set(rc={"figure.dpi": 600, 'savefig.dpi': 600})
sns.set_style(style="ticks")
sns.set_context("paper")
sns.set_palette('viridis')

ax = sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 6},
                 cmap="Blues", linewidths=0.25, linecolor="grey",
                 vmin=0, vmax=1, cbar_kws={"shrink": 0.7})
ax.set_xticklabels(ax.get_xticklabels(), size=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=7)

confusion_matrix.to_csv("confusion_matrix.csv")



# plot curves
y_prob = best_model.predict_proba(X_test)

n_classes = len(np.unique(y))
fpr_dict = dict()
tpr_dict = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(np.eye(n_classes)[y_test].ravel(), y_prob.ravel()) 
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Plot the micro-average ROC curve and ROC curves for all classes
sns.set(rc={'figure.figsize': (4, 3)}) #change where required
sns.set(rc={"figure.dpi": 600, 'savefig.dpi': 600})
sns.set_style(style="ticks")
sns.set_context("paper")
sns.set_context("paper", font_scale=1.6)

lw = 2

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, 
         label="micro-average ROC curve (AUC = {:.2f})".format(roc_auc["micro"]))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (class {}) (AUC = {:.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.8), fontsize="small", frameon=False)
plt.show()



#or just the micro-averaged ROC-CURVE
plt.plot(fpr["micro"], tpr["micro"], color="tab:blue", lw=2,
         label="micro-average ROC curve (AUC = {:.2f})".format(roc_auc["micro"]))
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.8), fontsize='small', frameon=False)
#plt.title("Micro-average ROC-AUC curve")
plt.text(0.6,0, "AUC = 1.00")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Micro-average ROC curve")



# Calculate precision, recall, and average precision for each class
precision_dict = dict()
recall_dict = dict()
average_precision_dict = dict()

for i in range(n_classes):
    precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test == i, y_prob[:, i])
    average_precision_dict[i] = average_precision_score(y_test == i, y_prob[:, i])



sns.set(rc={'figure.figsize': (8, 6)})
sns.set(rc={"figure.dpi": 600, 'savefig.dpi': 600})
sns.set_style(style="ticks")
sns.set_context("paper")
sns.set_context("paper", font_scale=2)



plt.figure()
# Plot micro-average precision-recall curve
precision_micro, recall_micro, _ = precision_recall_curve(np.eye(n_classes)[y_test].ravel(), y_prob.ravel())
average_precision_micro = average_precision_score(np.eye(n_classes)[y_test].ravel(), y_prob.ravel())
plt.plot(recall_micro, precision_micro, color="gold", lw=lw,
         label="micro-average PR curve (average precision = {:.2f})".format(average_precision_micro))

# Plot precision-recall curves for each class
for i in range(n_classes):
    plt.plot(recall_dict[i], precision_dict[i], lw=lw,
             label="PR curve (class {}) (average precision = {:.2f})".format(i, average_precision_dict[i]))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.8), fontsize="small", frameon=False)
plt.show()



# get the important features for Markov modelling
feature_importance = pd.DataFrame({"Feature": feature_columns,
                                   "Importance": best_model.feature_importances_},
                                  index=feature_columns).sort_values(
                                           by="Importance",
                                           ascending=False)
       
feature_importance.to_csv("ImportantFeatures.csv")                            
                                       
#plot the importance of each features
sns.set(rc={'figure.figsize':(3,6)})
sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
sns.set_style(style="ticks")
sns.set_context("paper", font_scale=1)
sns.set_palette('viridis')

ax = sns.barplot(data=feature_importance.head(50), x="Importance", y="Feature",
                 color="black", orient = "h") #change the numer of features shown as required

ax.set_ylabel("Feature",fontsize=14)
ax.set_xlabel("Importance",fontsize=14)

plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.yticks(size=7)
sns.despine()



# filter column data from original dataframe
top_20 = feature_importance.head(20)
top_20_indices = top_20["Feature"].to_list()

sel_feats = df[top_20_indices]
sel_feats.columns = columns
sel_feats["label"] = df["label"]

norm_features = pd.DataFrame(X_norm, columns=feature_columns)
sel_feats_norm = norm_features[top_20_indices]
sel_feats_norm.columns = columns
sel_feats_norm["label"] = df["label"]


sel_feats.to_csv("selected_features_raw.csv")
sel_feats_norm.to_csv("selected_features_Z-norm.csv")


