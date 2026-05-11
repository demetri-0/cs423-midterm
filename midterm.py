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

    return PCA, mlba, mo, pd, px


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


@app.cell
def _(df, px):
    heatmap_numeric_df = df.select_dtypes(include="number")
    heatmap_corr = heatmap_numeric_df.corr(numeric_only=True)

    fig1 = px.imshow(
        heatmap_corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap of Numeric Variables",
    )
    fig1.update_layout(width=900, height=800)
    fig1
    return


@app.cell
def _(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scatter_target = "Graduation rate"
    scatter_numeric_df = df.select_dtypes(include="number")
    scatter_corr_to_target = (
        scatter_numeric_df.corr(numeric_only=True)[scatter_target]
        .drop(labels=[scatter_target])
        .abs()
        .sort_values(ascending=False)
    )
    scatter_top_features = scatter_corr_to_target.head(3).index.tolist()

    fig2 = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=scatter_top_features,
    )

    for i, feature in enumerate(scatter_top_features, start=1):
        fig2.add_trace(
            go.Scatter(
                x=df[feature],
                y=df[scatter_target],
                mode="markers",
                marker=dict(size=6, opacity=0.6),
                name=feature,
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig2.update_xaxes(title_text=feature, row=1, col=i)
        fig2.update_yaxes(title_text=scatter_target, row=1, col=i)

    fig2.update_layout(
        title="Top 3 Features Most Correlated with Graduation Rate",
        width=1100,
        height=400,
    )
    fig2
    return


@app.cell
def _(df, px):
    fig3 = px.box(
        df,
        y="Graduation rate",
        title="Box Plot of Graduation Rate",
    )
    fig3.update_layout(width=700, height=500)
    fig3
    return


@app.cell
def _(df):
    numeric_only_df = df.select_dtypes(include="number")
    numeric_only_df = numeric_only_df.drop(columns=["Public (1)/ Private (2)"])
    numeric_only_df.shape
    return (numeric_only_df,)


@app.cell
def _(numeric_only_df):
    clean_numeric_df = numeric_only_df.dropna()
    clean_numeric_df.shape
    return (clean_numeric_df,)


@app.cell
def _(clean_numeric_df):
    clean_numeric_df.sample(5)
    return


@app.cell
def _(PCA, clean_numeric_df, pd):
    raw_pca_model = PCA()
    raw_pca_model.fit(clean_numeric_df)
    raw_pca_component_labels = [
        f"PC{i}" for i in range(1, len(raw_pca_model.explained_variance_ratio_) + 1)
    ]

    raw_pca_summary = pd.DataFrame(
        {
            "Component": raw_pca_component_labels,
            "Explained Variance Ratio": raw_pca_model.explained_variance_ratio_,
            "Cumulative Explained Variance": raw_pca_model.explained_variance_ratio_.cumsum(),
        }
    )
    raw_pca_summary
    return raw_pca_component_labels, raw_pca_model


@app.cell
def _(clean_numeric_df, pd, raw_pca_component_labels, raw_pca_model):
    raw_pca_loadings = pd.DataFrame(
        raw_pca_model.components_.T,
        index=clean_numeric_df.columns,
        columns=raw_pca_component_labels,
    )
    raw_pca_loadings
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based on the PCA analysis of the cleaned data above, it appears that only the first two principal components would be needed to appropriately reduce dimensionality given they explain >90% of the variance in the data. Upon further examining the feature weights in these components, it appears that PC1 encompasses tuition information, and PC2 encompasses a mixture of student quantities and tuition. It is important, however, to keep in mind that this initial analysis is based on data that is not normalized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Should the data be normalized?

    Yes, this data should be normalized. After cleaning, the range of the remaining features varies heavily, containing units such as dollars, student counts, and percentages. Normalization would bring all these values into the same scale, and would make any following analysis more meaningful.
    """)
    return


if __name__ == "__main__":
    app.run()
