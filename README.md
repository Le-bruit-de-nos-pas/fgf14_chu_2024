# Paulo Bastos   July 25th 2024 

```python
# pip install torch==2.3.0 torchvision==0.18.0
#import torch
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu, fisher_exact, stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

#from statsmodels.api import add_constant, OLS
#from statsmodels.gam.api import BSplines
#from statsmodels.gam.generalized_additive_model import GLMGam

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost
````

## Overall Data

```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
print(df.head())
````

```python
column_names = pd.DataFrame(df.columns, columns = ['Names'])
print(column_names.to_string(index=False))
````

```python
counts = []

for column in df.columns:
    num_unique_values = df[column].nunique()
    counts.append([column, num_unique_values])

counts = pd.DataFrame(counts, columns=['Variable', 'Number_Unique'])
print(counts.to_string(index=False))
````

```python
def categorize_var(number_unique):
    if number_unique <= 2:
        return "Binary"
    elif number_unique == 3:
        return "Group"
    else:
        return "Continuous"
    
counts['Variable_Type'] = counts['Number_Unique'].apply(categorize_var)
counts.at[0, 'Variable_Type'] = 'ID'
print(counts.to_string(index=False))
````

```python
continuous_vars = counts[counts['Variable_Type'] == 'Continuous']['Variable'].tolist()
binary_vars = counts[counts['Variable_Type'] == 'Binary']['Variable'].tolist()
print(df.shape, len(continuous_vars), len(binary_vars))  
````

```python
columns_to_subset = ['genetics_conclusion'] + continuous_vars
df_continuous = df[columns_to_subset].copy()
df_continuous[continuous_vars] = df_continuous[continuous_vars].apply(pd.to_numeric, errors='coerce')
df_continuous
````

```python
def calculate_stats_continuous(df, subset_column, subset_value):
    subset_df = df[df[subset_column] == subset_value]
    summary_stats_data = []

    for column in subset_df.columns:
        if pd.api.types.is_numeric_dtype(subset_df[column]):  
            mean = subset_df[column].mean()
            sd = subset_df[column].std()
            summary_stats_data.append({'Column': column, 'Mean': mean, 'SD': sd})

    summary_stats_df = pd.DataFrame(summary_stats_data)
    summary_stats_df['Mean ± SD'] = summary_stats_df.apply(lambda row: f"{row['Mean']:.2f} ± {row['SD']:.2f}", axis=1)
    summary_stats_df.drop(['Mean', 'SD'], axis=1, inplace=True)
    return summary_stats_df
````

```python
result_FGF = calculate_stats_continuous(df_continuous, 'genetics_conclusion', 'FGF')
result_Neg = calculate_stats_continuous(df_continuous, 'genetics_conclusion', 'Negative')
result_MSA = calculate_stats_continuous(df_continuous, 'genetics_conclusion', 'MSA')

result_FGF.to_csv('data/result_FGF_cont.csv')
result_Neg.to_csv('data/result_Neg_cont.csv')
result_MSA.to_csv('data/result_MSA_cont.csv')
````

## Summary Binary Features

```python
columns_to_subset = ['genetics_conclusion'] + binary_vars
df_binary = df[columns_to_subset].copy()
df_binary[binary_vars] = df_binary[binary_vars].apply(pd.to_numeric, errors='coerce')
df_binary
````

```python
def calculate_stats_binary(df, subset_column, subset_value):
    subset_df = df[df[subset_column] == subset_value]
    summary_stats_data = []

    for column in subset_df.columns:
        if pd.api.types.is_numeric_dtype(subset_df[column]) and subset_df[column].nunique() <= 2:
            sum_ = subset_df[column].sum()
            percentage = (sum_ / len(subset_df)) * 100
            summary_stats_data.append({'Column': column, 'Sum': sum_, 'Percentage': percentage})

    summary_stats_df = pd.DataFrame(summary_stats_data)
    summary_stats_df['n (%%)'] = summary_stats_df.apply(lambda row: f"{row['Sum']} ({row['Percentage']:.2f}%)", axis=1)
    summary_stats_df.drop(['Sum', 'Percentage'], axis=1, inplace=True)

    return summary_stats_df
````

```python
result_FGF = calculate_stats_binary(df_binary, 'genetics_conclusion', 'FGF')
result_Neg = calculate_stats_binary(df_binary, 'genetics_conclusion', 'Negative')
result_MSA = calculate_stats_binary(df_binary, 'genetics_conclusion', 'MSA')

result_FGF.to_csv('data/result_FGF_bin.csv')
result_Neg.to_csv('data/result_Neg_bin.csv')
result_MSA.to_csv('data/result_MSA_bin.csv')
````

## Mann Whitney U

```python
def perform_mann_whitney(df, continuous_columns, condition_column, g_1, g_2 ):
    results = []
    
    for col in continuous_columns:
        try:
            group_1 = df[df[condition_column] == g_1][col].dropna()
            group_2 = df[df[condition_column] == g_2][col].dropna()
            _, p_value = mannwhitneyu(group_1, group_2, alternative='two-sided')
            results.append((col, p_value.round(4)))

        except Exception as e:
            print(f"Error occurred for column '{col}': {e}")
            continue

    results_df = pd.DataFrame(results, columns=['Column', 'P-Value'])
    return results_df
````

```python
mann_whit_fgf_neg = perform_mann_whitney(df_continuous, continuous_vars, 'genetics_conclusion', 'FGF', 'Negative')
mann_whit_fgf_msa = perform_mann_whitney(df_continuous, continuous_vars, 'genetics_conclusion', 'FGF', 'MSA')

mann_whit_fgf_neg.to_csv('data/mann_whit_fgf_neg.csv')
mann_whit_fgf_msa.to_csv('data/mann_whit_fgf_msa.csv')
````

## Fisher's Exact

```python

def perform_fishers_exact(df, binary_columns, condition_column, g_1, g_2):
    results = []
    
    # Filter DataFrame for relevant groups
    filtered_df = df[(df[condition_column] == g_1) | (df[condition_column] == g_2)]

    for col in binary_columns:
        try:
            # Create contingency table and fill missing values with zeros
            contingency_table = pd.crosstab(filtered_df[condition_column], filtered_df[col]).reindex(
                index=[g_1, g_2], columns=[0, 1], fill_value=0
            )
            
            # Perform Fisher's exact test
            odds_ratio, p_value = fisher_exact(contingency_table)
            p_value = round(p_value, 4)
            
            # Append results
            results.append((col, p_value))
        except Exception as e:
            print(f"Error occurred for column '{col}': {e}")
            results.append((col, None))
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['Column', 'P-Value'])
    
    return results_df
````

```python
fisher_fgf_neg = perform_fishers_exact(df_binary, binary_vars, 'genetics_conclusion', 'FGF', 'Negative')
fisher_fgf_msa = perform_fishers_exact(df_binary, binary_vars, 'genetics_conclusion', 'FGF', 'MSA')

fisher_fgf_neg = fisher_fgf_neg.rename(columns={"P-Value": "P-Value_Neg"})
fisher_fgf_msa = fisher_fgf_msa.rename(columns={"P-Value": "P-Value_MSA"})
````

```python
merged_df = pd.merge(fisher_fgf_neg, fisher_fgf_msa, on='Column', how='left')
merged_df
merged_df.to_csv('data/merged_df.csv')
````



## XGBoost FGF+ vs FGF-

```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df.drop(['patient_id'], axis=1, inplace=True)
````

```python
missing_columns_bool = df.isnull().any()
````

```python
filtered_df = df[ (df['genetics_conclusion'] == 'FGF') | (df['genetics_conclusion'] == 'Negative') ]
filtered_df
````

```python
missing_columns_bool = df.isnull().any()
````

```python
filtered_df.isnull().sum().sort_values(ascending=False)
````

```python
filtered_df = filtered_df[['genetics_conclusion']  + binary_vars ].copy()
filtered_df.isnull().sum().sort_values(ascending=False)
````

```python
filtered_df.isnull().sum().sort_values(ascending=False)
````

```python
filtered_df['genetics_conclusion'] = filtered_df['genetics_conclusion'].apply(lambda x: 1 if x == 'FGF' else 0)
````

```python
y = filtered_df['genetics_conclusion']
X = filtered_df.drop(columns=['genetics_conclusion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````

```python
model = xgboost.XGBClassifier(n_estimators=142, max_depth=2, objective = "binary:logistic").fit(X, y)
model.fit(X_train, y_train)
````

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

````

```python
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap_values.display_data = X.values
````

```python
def rgba_to_hex(rgba):
    """Convert RGBA color code to hexadecimal color code."""
    r, g, b, a = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

cmap = plt.get_cmap('RdBu')
blue_hex = rgba_to_hex(cmap(0.0))
red_hex = rgba_to_hex(cmap(1.0))
print(red_hex, blue_hex)
````

```python

shap.initjs()
shap.force_plot(shap_values[28], plot_cmap=["#053061", "#67001F"])
#1,2,3,7, 11,15
#23,24,25;28,32,35,
````

```python
shap.plots.beeswarm(shap_values,  color=plt.get_cmap("RdBu"))
````

```python
shap.plots.heatmap(shap_values, max_display=10, plot_width=6, cmap=plt.get_cmap("RdBu"))
````

```python
shap.plots.heatmap(shap_values, max_display=10, plot_width=6, instance_order=shap_values.sum(1), cmap=plt.get_cmap("RdBu"))
````

```python
X = X.values
y = y.values
````

```python

cv = StratifiedKFold(n_splits=2)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='firebrick', label=f'Mean ROC (area = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='lightgray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
````




## XGBoost FGF+ vs FGF-


```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df.drop(['patient_id'], axis=1, inplace=True)
````

```python
missing_columns_bool = df.isnull().any()
````

```python
filtered_df = df[ (df['genetics_conclusion'] == 'FGF') | (df['genetics_conclusion'] == 'MSA') ]
filtered_df
````

```python
missing_columns_bool = df.isnull().any()
````

```python
filtered_df.isnull().sum().sort_values(ascending=False)
````

```python
filtered_df = filtered_df[['genetics_conclusion']  + binary_vars ].copy()
filtered_df.isnull().sum().sort_values(ascending=False)
````

```python
filtered_df['genetics_conclusion'] = filtered_df['genetics_conclusion'].apply(lambda x: 1 if x == 'FGF' else 0)
````

```python
y = filtered_df['genetics_conclusion']
X = filtered_df.drop(columns=['genetics_conclusion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````

```python
model = xgboost.XGBClassifier(n_estimators=142, max_depth=2, objective = "binary:logistic").fit(X, y)
model.fit(X_train, y_train)
````

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
````

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
````

```python
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
````

```python
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap_values.display_data = X.values
````

```python
def rgba_to_hex(rgba):
    """Convert RGBA color code to hexadecimal color code."""
    r, g, b, a = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

cmap = plt.get_cmap('RdBu') # cmap(#) returns list with rgba
blue_hex = rgba_to_hex(cmap(0.0))
red_hex = rgba_to_hex(cmap(1.0))
print(red_hex, blue_hex)
````

```python
shap.initjs()
shap.force_plot(shap_values[11], plot_cmap=["#053061", "#67001F"])
# 1,2,4,5,6;9;10,11;19
# 22 24 25 29 31
````

```python
shap.plots.beeswarm(shap_values,  color=plt.get_cmap("RdBu"))
````

```python
shap.plots.heatmap(shap_values, max_display=10, plot_width=6, cmap=plt.get_cmap("RdBu"))
````

```python
shap.plots.heatmap(shap_values, max_display=10, plot_width=6, instance_order=shap_values.sum(1), cmap=plt.get_cmap("RdBu"))
````


```python
np.exp(-3)/(1+np.exp(-3))
````


## Correlation between allele size and SARA score

```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df = df[df['genetics_conclusion'] == 'FGF']
df = df[['patient_id', 'allele_1', 'allele_2', 'SARA']]
df['max'] = df[['allele_1', 'allele_2']].max(axis=1, skipna=True)
df = df[['patient_id', 'max',  'SARA']]
````



```python
df.dropna()['SARA']
df.dropna()['SARA'].corr(df.dropna()['max']) # 0.01
stats.pearsonr(df.dropna()['SARA'], df.dropna()['max']) # -0.4
````



```python
df = df.dropna(subset=['SARA', 'max'])
df['SARA'] = df['SARA'].astype(int)
df['max'] = df['max'].astype(int)
````



```python
plt.figure(figsize=(8, 4))
sns.scatterplot(x='max', y='SARA', data=df, edgecolor='firebrick', marker='o', linewidth=2, facecolor='none')
sns.regplot(x='max', y='SARA', data=df, scatter=False, color='firebrick')
plt.xlabel('\n Allele Length')
plt.ylabel('SARA \n')
plt.title('Scatter plot \n Allele Length vs SARA score')
plt.grid(True)
plt.grid(False)
plt.show()
````

## Triad (Tetrad) Performance OLD Tetrad


```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df['onset_before_45'] = df[['age_epi_sympt']].min(axis=1, skipna=True)
df['onset_before_45'] = df['onset_before_45'].apply(lambda x: 1 if x < 45 else 0)
df['onset_before_45']
````


```python
df = df[['genetics_conclusion', 'onset_before_45', 'dx_dysarthria', 'dx_downbeat_nystagmus', 'dx_episodic_imbalance'] ]
````


```python
filtered_df = df[ (df['genetics_conclusion'] == 'FGF') | (df['genetics_conclusion'] == 'Negative') ].copy()
filtered_df['genetics_conclusion'] = filtered_df['genetics_conclusion'].apply(lambda x: 1 if x == 'FGF' else 0)
````



```python
y = filtered_df['genetics_conclusion']
X = filtered_df.drop(columns=['genetics_conclusion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````



```python
model = xgboost.XGBClassifier(n_estimators=142, max_depth=2, objective = "binary:logistic").fit(X, y)
model.fit(X_train, y_train)
````



```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
````



```python
X = X.values
y = y.values

cv = StratifiedKFold(n_splits=3)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='midnightblue', label=f'Mean ROC (area = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='lightgray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
````

## Triad (Tetrad) Performance USING OUR TOP 4

```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df['onset_before_45'] = df[['age_epi_sympt']].min(axis=1, skipna=True)
df['onset_before_45'] = df['onset_before_45'].apply(lambda x: 1 if x < 45 else 0)
````


```python
df = df[['genetics_conclusion', 'cerebellar_vermis_atrophy', 'dx_episodic_imbalance', 'dx_nystagmus',  'epi_sympt'] ]
````

```python
filtered_df = df[ (df['genetics_conclusion'] == 'FGF') | (df['genetics_conclusion'] == 'Negative') ].copy()
filtered_df['genetics_conclusion'] = filtered_df['genetics_conclusion'].apply(lambda x: 1 if x == 'FGF' else 0)
````

```python
y = filtered_df['genetics_conclusion']
X = filtered_df.drop(columns=['genetics_conclusion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
````

```python
model = xgboost.XGBClassifier(n_estimators=142, max_depth=2, objective = "binary:logistic").fit(X, y)
model.fit(X_train, y_train)
````

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
````

```python
X = X.values
y = y.values

cv = StratifiedKFold(n_splits=3)
tprs = []
aucs = []
mean_fpr_v2 = np.linspace(0, 1, 100)

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr_v2, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

mean_tpr_v2 = np.mean(tprs, axis=0)
mean_tpr_v2[-1] = 1.0
mean_auc_v2 = auc(mean_fpr_v2, mean_tpr_v2)

plt.plot(mean_fpr, mean_tpr, color='midnightblue', label=f'Mean ROC (area = {mean_auc:.2f})')
plt.plot(mean_fpr_v2, mean_tpr_v2, color='firebrick', label=f'Mean ROC (area = {mean_auc_v2:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='lightgray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
````

## Correlation between Allele Length and Age of Onset

```python
df = pd.read_csv('data/FGF14_pb_may_22.csv', delimiter=';', header=0)
df = df[df['genetics_conclusion'] == 'FGF']
df['onset'] = df[['age_epi_sympt']].min(axis=1, skipna=True)
df = df[['patient_id', 'allele_1', 'allele_2', 'onset']]
df['max_allele'] = df[['allele_1', 'allele_2']].max(axis=1, skipna=True)
df = df[['patient_id', 'max_allele',  'onset']]
````

```python
df['max_allele'].corr(df['onset']) # -0.3552673018765272
stats.pearsonr(df['max_allele'], df['onset']) # -0.4
````

```python
df['max_allele'] = df['max_allele'].astype(int)
df['onset'] = df['onset'].astype(int)


plt.figure(figsize=(8, 4))
sns.scatterplot(x='max_allele', y='onset', data=df, edgecolor='firebrick', marker='o', linewidth=2, facecolor='none')
sns.regplot(x='max_allele', y='onset', data=df, scatter=False, color='firebrick')
plt.xlabel('\n Allele Length')
plt.ylabel('Age at onset \n')
plt.title('Scatter plot \n Allele Length vs Age at onset')
plt.grid(True)
plt.grid(False)
plt.show()
````

```python
df['max_allele'] = df['max_allele'].astype(int)
df['onset'] = df['onset'].astype(int)

x_spline = df['max_allele']
splines = BSplines(x_spline, df=[4], degree=[3])
model = GLMGam(df['onset'], smoother=splines).fit()

x_pred = np.linspace(df['max_allele'].min(), df['max_allele'].max(), 100)
spline_pred = BSplines(x_pred, df=[4], degree=[3])
y_pred = model.predict(spline_pred.basis)
````

```python
plt.figure(figsize=(8, 4))
sns.scatterplot(x='max_allele', y='onset', data=df, alpha=0.8, edgecolor='firebrick', marker='o', linewidth=2, facecolor='none')
plt.plot(x_pred, y_pred, color='firebrick', linewidth=2)
plt.xlabel('\n Allele Length')
plt.ylabel('Age at onset \n')
plt.title('Cubic Spline Regression \n Allele Length vs Age at Onset')
plt.grid(True)
plt.grid(False)
plt.show()
````
