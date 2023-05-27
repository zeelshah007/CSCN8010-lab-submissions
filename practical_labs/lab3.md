## Univariate Linear Regression on the Scikit-Learn Diabetes Dataset

This lab should be submitted as a notebook and an HTML, in the same manner as lab2. Format your HTML as a short report. The focus should be the linear regression and reporting of its results. 

* `Scikit-Learn` provides toy datasets ([list](https://scikit-learn.org/stable/datasets/toy_dataset.html)).
* In this practical lab we will explore a univariate linear regression using the diabetes dataset. Here is a [description](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) of the dataset and its columns. **Make sure to go over this description, when you start exploring the data**.
* **The data can be loaded into a notebook by:**
```python
from sklearn import datasets
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
```
* Follow the relevant steps in the [Machine Learning Workflow](https://github.com/ageron/handson-ml3/blob/main/ml-project-checklist.md). Steps 1,2,3,6,7 are probably those that you would like to go over. Step 4 is probably not required, given the linear model being used. No need to explore the whole dataset. For data exploration, keep it short 1-3 graphs. The focus is the linear regression, and a presentation of its results.
* Run a univariate linear regression between `bmi body mass index` (independent variable) and `s6 glu, blood sugar level` (dependent variable).
* Report your results: the model's parameters (slope and intercept), the loss on the train and test datasets, graphs of the training data, test data and the model. Provide your take-aways from the fit. Is it a good one? 


### Notes

1. While partitioning the data to train and test is required for reporting the results on unseen-data-during-training, a validation set is not required. That is, unless you intend to train various models, or test various hyperparameters. These is not an expectation, so most likely a validation set is not needed.