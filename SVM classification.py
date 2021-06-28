# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:48:08 2021

@author: EricH
"""

import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url= HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
    

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

load_housing_data()
    
housing = load_housing_data()
housing.info()
housing.describe()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

len(test_set)

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2*32
def split_train_test_by_id(data, test_ratio, id_column):
   ids = data[id_column]
   in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
   return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0.,1.5,3.0,4.5,6.,np.inf],
                              labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#see if it works as expected
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#removing income category attribute so data is back to its pre-operative state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.copy()
# plot the data geographically
housing.plot(kind="scatter", x="longitude", y="latitude")

# try to use the datapoints as shading instead
housing.plot(kind="scatter", x="longitude", y="latitude", alpha = .1)
#plot it together using population and lat
housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.4,
            s = housing["population"]/100, label= "population", figsize = (10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
plt.legend()


# find correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#use scatter matrix to capture all relationships
from pandas.plotting import scatter_matrix

thingstoscatter = ["median_house_value", "median_income", "total_rooms",
                   "housing_median_age"]
scatter_matrix(housing[thingstoscatter], figsize=(13,7))

#zoom in on median income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=.15)

#create new attributes that are more representative of home value
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["pop_per_household"] = housing["population"]/housing["households"]

#clean up the data set for new operations
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.fit(housing_num)

housing_tr = pd.DataFrame(X, columns = housing_num.columns,
                          index = housing_num.index)
housing_cat = housing[["ocean_proximity"]]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
# fit the imputer to the dataset
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

ordinal_encoder.categories_
# one hot encoder allows you to get the missing values in a categorical variable
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
# convert the encoder to a sparse array in numpy
housing_cat_1hot.toarray()

cat_encoder.categories_

# now we need to create a transformer.
# needs to work with pipelines and duck typing.

# need to create a class and implement three methods
# fit returning self, transform(), fit_transform()

#create small transformer class that adds previous combined attributes

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def _init_(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self # nothing else to do
    
    def transform(self,X):
        rooms_per_household = X[:, rooms_ix] / X[:,households_ix]
        population_per_household = X[:, population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


#need feature scaling. algos dont perform well when numerical attributes have different scales

#two ways to get attributes to have the same scale: min max scaling and standardization
#you MUST fit the scalars to the training data, not to the full dataset.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScalar

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "Median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScalar()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Another way to transform the data as long as its in pandas DF

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
#constructor requires a list of tuples
housing_prepared = full_pipeline.fit_transform(housing)

#finally ready to select and train a model

from sklearn.linear_model import LinearRegression

# start with linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)



some_data = housing.iloc[5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# check RMSE on the whole training set

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# rmse is really big, almost half of the training data average. This is underfitting.

#instead, try a decision tree regressor, can find nonlinear relationships in data.

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#Then evaluate on training set
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_mse

#overfit the training data! Now we have to redo some of our cross validation

#use Scikit's K-fold cross validation, randomly splits the data and then uses
# gives you 10 different scores depending on which fold was the best

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring = "neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Dev.:", scores.std())
    












