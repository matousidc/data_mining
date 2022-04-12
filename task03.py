import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report


pd.set_option('max_columns', None)
pd.options.display.width = None
data = sns.load_dataset('titanic')
data = data.drop(['who', 'adult_male', 'embark_town', 'alive', 'class', 'alone', 'parch', 'sibsp', 'deck', 'embarked'],
                 axis='columns')
X = data.drop(['survived'], axis='columns')
y = data['survived']
print(data)

# cleaning data
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("scaler", StandardScaler())])

categorical_features = ["sex"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)])

# training
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
clf = Pipeline(steps=[('preprocesor', preprocessor),
                      ('classifier', GaussianNB())])
clf.fit(x_train, y_train)

# results
score = clf.score(x_test, y_test)
print('score:', score)
pred = clf.predict(x_test)
matrix = confusion_matrix(y_test, pred)
sns.heatmap(matrix/np.sum(matrix), square=True, annot=True, fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=['predicted 0', 'predicted 1'], yticklabels=['actual 0', 'actual 1'])
print(classification_report(y_test, pred))
plt.show()

