import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Midterm Exam: Business Analytics / Data Analytics
    # Spring 2026

    #### Answer questions in cells below the question
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import plotly.express as px
    import mlba
    import numpy as np
    from sklearn import preprocessing
    from sklearn.decomposition import PCA 

    return mlba, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Explain the difference between variance and bias as sources of generalization errors in machine learning?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bias occurs when a model poorly captures trends in the data, resulting in underfitting that reduces its predictive power. Variance occurs when a model fits the training set of data too well, and becomes very sensitive to deviations from the training set. A model with heavy bias typically has high training and test error, while a model with heavy variance typically has low training error and high test error. Bias and variance are inversely related - as one goes up, the other comes down. The goal of a strong model is to minimize both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. If both train and test set RMSE are high what can we say about bias vs. variance?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the training and test sets have high RMSE, we can infer that the model has high bias. The poor performance in the training set confirms that the model is not properly capturing trends in the data, which is a clear sign of underfitting and the presence of bias. High bias is the main problem for this model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Explain the curse of dimensionality? How can that problem be addressed in machine learning?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curse of dimensionality refers to the problems that arise in a model when the examined data has a large number of dimensions/features. These problems include overfitting and heavy computational load. When there are many dimensions, data becomes sparse, and it becomes difficult to capture meaningful trends. These problems can be tackled through dimension reduction and feature selection. Through examining the correlation among features and the target, the least meaningful features can be dropped. A technique like PCA can combine features to reduce dimensionality while also keeping any existing trends in the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. If you want to compare the distributions of five variables what visualization would you use and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Box plots would be my choice to visualize the distribution of five variables. Box plots let viewers immediatly see the minimum, Q1, median, Q2, and maximum of a feature, and are relatively easy to interpret. Five box plots side-by-side would be my preferred way of examining five variables and their distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5. The dataset Universities.csv on American college and university rankings contains information on 1,302 American colleges and universities offering an undergraduate program. For each university there are 17 measurements that include continous measurements (such as tuition and graduation rates) and categorical measurements (such as location by state and whether it is a private or public school)
    - Conduct a visual exploratory analysis of this dataset and discuss your key findings.
    - Remove all categorical variables from the datset
    - Remove all missing numerical values
    - Conduct a Principal Components Analysis on the cleaned data and discuss your findings.
    - Should the data be normalized? why?
    - Normalize the data and re-run your PCA?
    - Discuss how many components you would keep and why? Can you interpret these components?

    **Note**: Each part of your analysis must be in a different code or markdown cell.
    """)
    return


@app.cell
def _(mlba):
    df = mlba.load_data('Universities.csv')
    return (df,)


@app.cell
def _(df):
    df.sample(5)
    return


@app.cell
def _(df):
    df.shape
    return


if __name__ == "__main__":
    app.run()
