## Practical Lab 4 - Multivariate Linear and Polynomial Regression, and Evaluation using MAE and MAPE.

_Note: The submission should be done as a notebook inside your `/students_submissions/STUDENT-ID/` folder, and as an HTML file, accessible online. Please provide the link to the HTML in the PR (pull request) description._

In this lab, we will run multivariate linear and polynomial regression, and evaluate them using the Mean Absolute Error ([MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)), and Mean Absolute Percentage Error ([MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)) metrices. The dataset will be Scikit Learn Diabetes dataset - the same one used in Practical Lab 3.

1. Use the code provided below to get the data, and run the train-test split. Description of each column can be found in [sklearn documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset). Look at the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes) for the `load_diabetes` method to know what are `as_frame` and `scaled` for.
2. Run a multivariate linear regression on all variables **(1 point)**
3. Run a multivariate polynomial regression of the 2nd degree on all variable (Hint: set `include_bias=False` in `PolynomialFeatures`) **(2 points)**
4. Compare the two models by looking at MAE and MAPE using the provided function
5. You can add any further comparisons and code (this is not necessary for a perfect score, but will be reviewed and evaluated)

At the end of your submission file, please answer the following questions **(3 points)**:
1. How many parameters are we fitting for in linear regression and in the polynomial regression? Explain these values. Hint: for explaining the parameters of the polynomial regression, you can use `poly.get_feature_names_out()`.
2. Which model would you choose for deployment, and why?
3. What insights regarding the goodness-of-fit can you draw from the graphs provided by the function below, for each of the models?

## Started Code 
_Copy this code to your notebook, and use it as your starting point._

```python
from sklearn import datasets
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(as_frame=True, scaled=False, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_estimation(model, y_train, y_test, y_pred_train, y_pred_test):
    """
    Evaluate the model using MAE. 
    Inputs:
        model: the model to be evaluated
        y_train: the true y values of the train dataset
        y_test: the true y values of the test dataset
        y_pred_train: the predicted y values of the train dataset
        y_pred_test: the predicted y values of the test dataset
    Returns:
        None. 
        Print the MAE of the train and test datasets and the equation of the model.
        Plots of the residuals, percentage error, and histogram of the percentage error.
    """
    # create three subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # plot a scatter plot of the residuals
    ax1.scatter(y_test, y_test - y_pred_test)
    ax1.set_xlabel('$y_{test}$')
    ax1.set_ylabel('Residual $*(y_{test} - y_{pred}) / y_{test})$')
    ax1.set_title('Residuals')
    # plot a scatter plot of the percentage error
    ax2.scatter(y_test, 100. * (y_test - y_pred_test) / y_test)
    ax2.set_xlabel('$y_{test}$')
    ax2.set_ylabel('percentage error $100*(y_{test} - y_{pred}) / y_{test})$')
    ax2.set_title('Percentage error (%)')
    # plot a histogram of the percentage error
    ax3.hist(100. * (y_test - y_pred_test) / y_test, bins=20)
    ax3.set_xlabel('percentage error $100*(y_{test} - y_{pred}) / y_{test})$')
    ax3.set_ylabel('Frequency (number of test samples)')
    ax3.set_title('Histogram of percentage error (%)')

    print(f'Test Dataset: Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.2f}')
    print(f'Train Dataset: Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.2f}')

    # print the equation
    print('y = ', end='')
    for i, coef in enumerate(model.coef_):
        print(f'{coef:.2f} * x_{i} + ', end='')
    print(f'{model.intercept_:.2f}')
    fig.tight_layout()

```