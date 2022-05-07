import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = sns.load_dataset('titanic')
data = data.drop(['who', 'adult_male', 'embark_town', 'alive', 'class', 'alone', 'parch', 'sibsp', 'deck', 'embarked'],
                 axis='columns')
data = data.dropna()

# k-nearest neighbours
data = data.drop(['fare'], axis='columns')
# cleaning data
numeric_features = ["survived", "age", "pclass"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("scaler", StandardScaler())])

categorical_features = ["sex"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)])

cleaned_data = preprocessor.fit_transform(data)
clf = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1).fit(cleaned_data)
passenger_index = random.randint(0, len(data))
distances, indices = clf.kneighbors(cleaned_data[passenger_index].reshape(1, -1), n_neighbors=5)
result = pd.DataFrame(columns=['pclass', 'sex', 'age', 'distance'])
for k in range(len(indices)):
    for (i, j) in zip(indices[k], distances[k]):
        temp = data.iloc[i].copy()
        temp['distance'] = j
        result = result.append(temp)
print('the passenger:')
print(data.iloc[passenger_index:passenger_index+1])
print("the neighbors:")
print(result)

# k-nearest nighbors classification
x = data.drop(['survived'], axis='columns')
y  = data['survived'] 
# cleaning data
numeric_features = ["age", "pclass"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("scaler", StandardScaler())])

categorical_features = ["sex"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)])
# training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())])
clf.fit(x_train, y_train)
# results
score = clf.score(x_test, y_test)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix/np.sum(matrix), annot=True, square=True,  fmt='.2%', cbar=False,
            xticklabels=['predicted 0', 'predicted 1'], yticklabels=['actual 0', 'actual 1'])
plt.show()
