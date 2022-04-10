import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report

pd.set_option('max_columns', None)
pd.options.display.width = None
'''data = sns.load_dataset('titanic')

data = data.drop(['who', 'adult_male', 'embark_town', 'alive', 'class', 'alone'], axis='columns')
X = data.drop(['survived'], axis='columns')
y = data['survived']

# cleaning the dataset
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                      ("scaler", StandardScaler())])

categorical_features = ["embarked", "sex", "deck"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# training
clf = Pipeline(steps=[("preprocessor", preprocessor),
                      ("classifier", LogisticRegression())])
fit_model = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
y_pred = clf.predict(x_test)

# print('score:', score)
# print('mean squared error:', mean_squared_error(y_test, y_pred))
# print('r2 score:', r2_score(y_test, y_pred))'''

# ******************************************************* cviko 5
'''sns.set_theme()
tips = sns.load_dataset('tips')
tips['ratio'] = tips['tip'] / tips['total_bill']
mean = tips.groupby('sex')['ratio'].mean()

# sns.lmplot(data=tips, x="total_bill", y="ratio", hue='sex')
sns.relplot(data=tips, x="total_bill", y="ratio", hue='sex')
plt.axhline(mean['Male'], 0, 50)
plt.axhline(mean['Female'], 0, 50, color='orange')
# plt.show()'''

# ************************************************ cviko 7
data = sns.load_dataset('titanic')
data_2 = data.copy()
data = data.drop(['who', 'adult_male', 'embark_town', 'alive', 'class', 'alone'], axis='columns')
print(data)
data = pd.concat([data, pd.get_dummies(data["sex"]),
                  pd.get_dummies(data["embarked"]),
                  pd.get_dummies(data["deck"])], axis=1)
data = data.drop(['sex', 'embarked', 'deck'], axis='columns')
data = data.fillna(method='bfill')
print(data)
x = data.drop(['survived'], axis='columns')
y = data['survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
clf = tree.DecisionTreeClassifier().fit(x_train, y_train)
fig, axes = plt.subplots(dpi=300)
tree.plot_tree(clf, feature_names=list(data),
               class_names=["survived", "died"],
               filled=True)

print(classification_report(y_test, clf.predict(x_test)))
# **********************
x = data_2.drop(['survived'], axis='columns')
y = data_2['survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                                      ("scaler", StandardScaler())])

categorical_features = ["embarked", "sex", "deck"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)])

clf = Pipeline(steps=[("preprocessor", preprocessor),
                      ("classifier", tree.DecisionTreeClassifier())])
clf.fit(x_train, y_train)
print(classification_report(y_test, clf.predict(x_test)))




