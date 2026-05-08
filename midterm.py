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
    2. If both train and test set RMSE are high what can we say about bias vs. variance?
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
    4. If you want to compare the distributions of five variables what visualization would you use and why?
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
