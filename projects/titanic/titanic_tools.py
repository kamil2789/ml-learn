import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def plot_people_in_each_class(titanic_data: pd.DataFrame):
    pclass = titanic_data.Pclass.value_counts().sort_index()
    pclass.plot(kind='bar')
    plt.xlabel("Class")
    plt.ylabel("People")
    plt.title("Number of people in each class")
    plt.show()


def plot_who_survived(titanic_data: pd.DataFrame):
    counts = titanic_data.groupby(["Pclass"])["Survived"].value_counts()
    counts.plot(kind="bar", color=["red", "green"])
    plt.xlabel("Class")
    plt.ylabel("People")
    plt.title("Number of people in each class who survived & not survived")
    plt.show()


def encode_data(titanic_data: pd.DataFrame):
    onehot = OneHotEncoder(sparse_output=False)
    encoded_data = pd.DataFrame(onehot.fit_transform(titanic_data[["Sex", "Embarked"]]))
    encoded_data.columns = onehot.get_feature_names_out(["Sex", "Embarked"])
    titanic_data = pd.concat([titanic_data, encoded_data], axis=1)
    titanic_data.drop(labels=["Sex", "Embarked"] , axis="columns", inplace=True)
    return titanic_data


def get_features(titanic_data: pd.DataFrame):
    return titanic_data.drop(labels=["Survived"] , axis="columns")


def get_labels(titanic_data: pd.DataFrame):
    return titanic_data["Survived"]


class MissingValuesHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def fill_age(self):
        median = self.data.Age.median()
        self.data["Age"].fillna(median, inplace=True)

    def fill_embarked(self):
        self.data["Embarked"].fillna("S", inplace=True)

    def fill_fare(self):
        mean = self.data.Fare.mean()
        self.data["Fare"].fillna(mean, inplace=True)

    def drop_columns(self):
        self.data.drop(labels=["Cabin", "Ticket", "Name", "PassengerId"], axis=1, inplace=True)