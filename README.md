# Business Analytics for Predicting High-Performing Job Listings

## Overview

This project applies **Business Analytics and Natural Language
Processing (NLP)** to LinkedIn job listings to investigate which
characteristics of job postings are associated with higher candidate
interest.

The central business question is:

> **Which job-listing characteristics are associated with higher
> candidate interest, and can the text of a job description be used to
> predict whether a listing will be classified as high-performing?**

The project covers the complete workflow from data collection and
preparation through exploratory analysis, TF-IDF feature extraction,
Logistic Regression classification, and model evaluation.

------------------------------------------------------------------------

## Business Problem

Recruiters and organizations publish large numbers of job listings, but
not every listing attracts the same level of candidate interest.

A job listing can differ in:

-   Job title
-   Job category
-   Company information
-   Location
-   Experience level
-   Employment type
-   Industry/sector
-   Job description
-   Salary information
-   Number of applicants

Understanding these patterns can help recruitment teams identify
characteristics associated with stronger candidate engagement and
improve future job-posting strategies.

This project therefore treats **candidate application activity as the
performance signal** for a job listing.

------------------------------------------------------------------------

# 1. Project Objectives

The project has five main objectives:

1.  Collect LinkedIn job-listing data programmatically.
2.  Clean and prepare the scraped data for analysis.
3.  Define a binary target representing high- and low-performing
    listings.
4.  Analyze textual characteristics of job descriptions.
5.  Build a machine-learning model using **TF-IDF + Logistic
    Regression** to predict listing performance.

------------------------------------------------------------------------

# 2. Data Collection

## Source

The project uses job-listing information collected from **LinkedIn job
postings**.

The data-collection workflow used **Apify**, a web-scraping and
automation platform, to collect structured job-listing information.

The scraped records contain fields such as:

-   Job title
-   Company name
-   Location
-   Job description
-   Experience level
-   Contract type
-   Job function
-   Sector
-   Company size
-   Company employee count
-   Company follower count
-   Company information
-   Salary information
-   Application count
-   Publication information
-   LinkedIn job identifier

## Apify Collection Workflow

The general collection process was:

``` text
LinkedIn Job Search
        |
        v
Define target job roles / search inputs
        |
        v
Run LinkedIn job scraper through Apify
        |
        v
Collect structured job-listing records
        |
        v
Export scraper output as CSV
        |
        v
datasetLinkedInraw.csv
```

The project targeted multiple technology, analytics, business,
engineering, and management roles. The role set used during collection
included:

-   Data Analyst
-   Data Scientist
-   Business Analyst
-   Data Engineer
-   Software Engineer
-   Machine Learning Engineer
-   AI Engineer
-   Cloud Engineer
-   DevOps Engineer
-   Cybersecurity Analyst
-   Product Manager
-   Financial Analyst
-   Marketing Analyst
-   HR Analyst
-   Operations Analyst
-   Sales Analyst
-   Project Manager
-   UI/UX Designer

The scraper output was exported to CSV and used as the input to the
preprocessing pipeline.

------------------------------------------------------------------------

# 3. Dataset

## Raw Dataset

File:

``` text
data/datasetLinkedInraw.csv
```

Current shape:

``` text
10,000 rows × 49 columns
```

The raw working file contains the original job-listing fields together with the project target.

Important columns include:

| Column | Description |
|---|---|
| `id` | Job-listing identifier |
| `title` | Job title |
| `companyName` | Company name |
| `location` | Job location |
| `description` | Job description |
| `applicationsCount` | LinkedIn application-count category/value |
| `experienceLevel` | Required experience level |
| `contractType` | Contract/employment type |
| `jobFunction` | Job function |
| `sector` | Sector |
| `companySize` | Company-size category |
| `companyEmployeeCount` | Estimated company employee count |
| `companyFollowerCount` | Company follower count |
| `companyFoundedYear` | Company founding year |
| `companyType` | Company type |
| `companyIndustries` | Company industries |
| `salary` | Salary information where available |
| `salaryMin` | Minimum salary where available |
| `salaryMax` | Maximum salary where available |
| `publishedAt` | Publication timestamp |
| `postedTime` | Posted-time information |
| `high_performing` | Binary target |
------------------------------------------------------------------------

# 4. Target Variable

The project uses:

``` text
high_performing
```

as the binary classification target.

The target has two classes:

``` text
0 = Low-performing
1 = High-performing
```

The target represents whether a job listing meets the project's
definition of higher candidate interest based on its application-count
information.

## Application-count categories

The LinkedIn data contains several forms of application information,
including examples such as:

``` text
Over 200 applicants
Be among the first 25 applicants
33 applicants
43 applicants
100 applicants
...
```

The preprocessing workflow distinguishes:

-   Censored high-count listings
-   Censored low-count listings
-   Exact numeric application counts
-   Unknown values

The project uses the applicant-count information to create the binary
target rather than treating the raw application-count text as a model
feature.

### Leakage prevention

The following application-count fields are not used as predictive input
to the text model:

``` text
applicationsCount
applications_numeric
applications_raw
applications_type
```

This is important because the target itself is derived from applicant
activity. Feeding the same information directly into the model would
create target leakage.

------------------------------------------------------------------------

# 5. Data Cleaning

The preprocessing notebook is:

``` text
data/dataset_processing.ipynb
```

The cleaning process is implemented through a reusable function:

``` python
def clean_data(input_csv, output_csv):
    ...
```

## 5.1 Load the dataset

The raw CSV is loaded with Pandas:

``` python
df_clean = pd.read_csv(input_csv)
```

## 5.2 Remove duplicate job listings

Duplicate listings are removed using:

``` python
["title", "companyName", "location"]
```

The first occurrence is retained.

This reduces repeated listings that could otherwise distort descriptive
statistics and model training.

## 5.3 Remove unnecessary scraper/metadata fields

The preprocessing notebook removes fields that are not required for the
analytical workflow, including:

``` text
workType
benefits
posterProfileUrl
posterFullName
scrapingInfo/index
scrapingInfo/page
scrapingInfo/title
scrapingInfo/inputUrl
descriptionHtml
companyLogo
applyUrl
jobUrl
companyUrl
```

Only columns that exist in the dataset are dropped, making the cleaning
function robust to small changes in the scraper output.

## 5.4 Save the cleaned dataset

The cleaned data is written to:

``` text
data/datasetLinkedInprocessed.csv
```

------------------------------------------------------------------------

# 6. Processed Dataset Used by the NLP Model

The analysis notebook loads:

``` python
file_path = "data/datasetLinkedInprocessed.csv"
df = pd.read_csv(file_path)
```

The current processed file contains:

``` text
10,000 rows × 3 columns
```

with:

``` text
id
description
high_performing
```

This is intentionally simplified for the text-classification stage.

The three fields are:

### `id`

Unique identifier for the listing/record.

### `description`

Full job-description text used as the primary NLP input.

### `high_performing`

Binary target:

``` text
0 = Low-performing
1 = High-performing
```

------------------------------------------------------------------------

# 7. Exploratory Data Analysis

The main analysis notebook is:

``` text
analysis.ipynb
```

The analysis starts with basic data-quality checks.

These include:

-   Dataset dimensions
-   Column names
-   Data types
-   Missing values
-   Duplicate rows
-   Duplicate IDs
-   Target distribution

------------------------------------------------------------------------

## 7.1 Text Feature Engineering

Three descriptive text features are created from the job description:

### Description length

``` python
df["description_length"] = (
    df["description"].str.len()
)
```

This measures the number of characters in the job description.

### Word count

``` python
df["description_word_count"] = (
    df["description"].str.split().str.len()
)
```

This measures the number of whitespace-separated words.

### Sentence count

``` python
df["description_sentence_count"] = (
    df["description"].str.count(r"[.!?]")
)
```

This provides a simple estimate of the number of sentences.

These features are used for descriptive analysis before the main NLP
model.

------------------------------------------------------------------------

# 8. Exploratory Visualizations

The analysis notebook creates several visualizations.

## Target distribution

A bar chart compares:

-   Low-performing listings
-   High-performing listings

## Job-description length

A histogram/KDE plot examines description length across the two
performance classes.

## Word count

A box plot compares job-description word counts between:

-   Low-performing
-   High-performing

These visualizations help identify whether basic text-length
characteristics differ between the two groups.

------------------------------------------------------------------------

# 9. TF-IDF Text Representation

The predictive model uses **TF-IDF (Term Frequency--Inverse Document
Frequency)** to transform job descriptions into numerical features.

The model uses the job-description text:

``` python
X = text_df["description"]
```

and the binary target:

``` python
y = text_df["high_performing"]
```

## TF-IDF configuration

The project uses:

``` python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True
)
```

### Parameter explanation

  Parameter               Value Purpose
  ----------------- ----------- -------------------------------------------------
  `lowercase`            `True` Converts text to lowercase
  `strip_accents`     `unicode` Normalizes accented characters
  `ngram_range`        `(1, 2)` Uses unigrams and bigrams
  `min_df`                  `3` Ignores extremely rare terms
  `max_df`               `0.95` Removes terms appearing in almost all documents
  `max_features`       `20,000` Limits vocabulary size
  `sublinear_tf`         `True` Applies logarithmic term-frequency scaling

Using unigrams and bigrams allows the model to learn from both
individual terms and short phrases.

For example:

``` text
data
machine learning
software engineer
project management
cloud computing
```

can all become model features.

------------------------------------------------------------------------

# 10. Train/Test Split

The dataset is divided into training and testing sets using:

``` python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The configuration means:

-   80% training data
-   20% testing data
-   Random seed: `42`
-   Stratified sampling

Stratification preserves approximately the same target-class
distribution in both sets.

------------------------------------------------------------------------

# 11. Machine Learning Model

The project uses:

> **TF-IDF + Logistic Regression**

The implementation uses a Scikit-learn Pipeline:

``` python
tfidf_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=20000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42
        )
    )
])
```

The pipeline ensures that the TF-IDF vocabulary is learned from the
training data as part of the model-fitting process.

------------------------------------------------------------------------

# 12. Logistic Regression

Logistic Regression is used as the binary classifier.

The model predicts:

``` text
0 → Low-performing
1 → High-performing
```

The probability of the positive class is also generated:

``` python
y_prob = tfidf_model.predict_proba(X_test)[:, 1]
```

These probabilities are required for ROC-AUC evaluation.

------------------------------------------------------------------------

# 13. Model Evaluation

The model is evaluated using several complementary metrics.

## Accuracy

Measures the proportion of test observations classified correctly.

``` text
Accuracy = Correct Predictions / Total Predictions
```

## Precision

Measures how many listings predicted as high-performing are actually
high-performing.

``` text
Precision = TP / (TP + FP)
```

## Recall

Measures how many actual high-performing listings are successfully
identified.

``` text
Recall = TP / (TP + FN)
```

## F1 Score

The harmonic mean of precision and recall.

``` text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

## ROC-AUC

Measures the model's ability to distinguish between the two classes
across classification thresholds.

The notebook calculates:

``` python
roc_auc_score(y_test, y_prob)
```

------------------------------------------------------------------------

# 14. Classification Report

The notebook also produces a Scikit-learn classification report
containing:

-   Precision
-   Recall
-   F1-score
-   Support

for both:

``` text
Low-performing
High-performing
```

This gives a class-level view of model performance rather than relying
only on overall accuracy.

------------------------------------------------------------------------

# 15. Confusion Matrix

A confusion matrix is generated to show:

-   True Negatives
-   False Positives
-   False Negatives
-   True Positives

This helps identify the types of classification errors made by the
model.

------------------------------------------------------------------------

# 16. ROC Curve

The notebook calculates the ROC curve using:

``` python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
```

The resulting plot shows the relationship between:

-   False Positive Rate
-   True Positive Rate

The area under the curve is reported as ROC-AUC.

------------------------------------------------------------------------

# 17. TF-IDF Term Analysis

The project also examines which TF-IDF terms have stronger positive or
negative Logistic Regression coefficients.

The learned vocabulary is obtained using:

``` python
feature_names = np.array(
    tfidf.get_feature_names_out()
)
```

The Logistic Regression coefficients are then paired with the
corresponding terms.

Terms with larger positive coefficients are more strongly associated
with the model's high-performing class, while terms with more negative
coefficients are associated with the low-performing class.

This provides an interpretable NLP component for the business analysis.

> **Important:** These coefficients indicate statistical associations
> learned by the model. They should not be interpreted as proof that an
> individual word or phrase causes higher candidate interest.

------------------------------------------------------------------------

# 18. Project Workflow

The complete workflow can be summarized as:

``` text
                DATA COLLECTION
                     |
                     v
       LinkedIn Job Listings via Apify
                     |
                     v
          datasetLinkedInraw.csv
                     |
                     v
              DATA CLEANING
                     |
                     +--> Remove duplicates
                     |
                     +--> Remove unnecessary fields
                     |
                     v
       datasetLinkedInprocessed.csv
                     |
                     v
          DATA QUALITY CHECKS
                     |
                     v
          TEXT FEATURE ENGINEERING
                     |
                     +--> Description length
                     +--> Word count
                     +--> Sentence count
                     |
                     v
              EXPLORATORY DATA
                 ANALYSIS
                     |
                     v
              TRAIN / TEST SPLIT
                     |
                     v
               TF-IDF VECTORIZER
                     |
                     v
            LOGISTIC REGRESSION
                     |
                     v
                PREDICTIONS
                     |
                     v
        MODEL EVALUATION & INTERPRETATION
                     |
          +----------+----------+
          |          |          |
       Accuracy    F1       ROC-AUC
          |          |          |
          +----------+----------+
                     |
                     v
             BUSINESS INSIGHTS
```

------------------------------------------------------------------------

# 19. Repository Structure

The project currently follows this structure:

``` text
FinalFolder/
│
├── README.md
│
├── analysis.ipynb
│
├── data/
│   ├── datasetLinkedInraw.csv
│   ├── datasetLinkedInprocessed.csv
│   └── dataset_processing.ipynb
│
└── .ipynb_checkpoints/
```

### `analysis.ipynb`

Main analytics and machine-learning notebook.

Contains:

-   Data loading
-   Data-quality checks
-   Text feature engineering
-   EDA
-   TF-IDF
-   Train/test split
-   Logistic Regression
-   Evaluation
-   Confusion matrix
-   ROC curve
-   TF-IDF term analysis

### `data/dataset_processing.ipynb`

Data-preparation notebook.

Contains:

-   Project paths
-   Raw-data loading
-   Duplicate removal
-   Unnecessary-column removal
-   Processed-data export

### `data/datasetLinkedInraw.csv`

Raw working dataset exported/assembled for the project.

### `data/datasetLinkedInprocessed.csv`

Processed text-classification dataset containing:

``` text
id
description
high_performing
```

------------------------------------------------------------------------

# 20. Technologies Used

## Programming Language

-   Python

## Data Analysis

-   Pandas
-   NumPy

## Visualization

-   Matplotlib
-   Seaborn

## Machine Learning

-   Scikit-learn

## NLP

-   TF-IDF Vectorization

## Classification

-   Logistic Regression

## Data Collection

-   Apify
-   LinkedIn job-listing data

## Development Environment

-   Jupyter Notebook

------------------------------------------------------------------------

# 21. Python Libraries

The main libraries used in the project are:

``` python
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
```

------------------------------------------------------------------------

# 22. Reproducibility

The project uses fixed random seeds where applicable.

The train/test split uses:

``` python
random_state=42
```

and Logistic Regression also uses:

``` python
random_state=42
```

This makes the train/test partition and model initialization
reproducible under the same software and data conditions.

------------------------------------------------------------------------

# 23. How to Run the Project

## Step 1 --- Clone/download the project

Place the project folder on your computer.

## Step 2 --- Install dependencies

Install the required Python packages:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Step 3 --- Open the project

From the project directory:

``` bash
jupyter notebook
```

or:

``` bash
jupyter lab
```

## Step 4 --- Run data preprocessing

Open:

``` text
data/dataset_processing.ipynb
```

Run the cells to:

1.  Load the raw CSV.
2.  Remove duplicate listings.
3.  Remove unnecessary columns.
4.  Save the processed dataset.

## Step 5 --- Run the analysis

Open:

``` text
analysis.ipynb
```

Run the notebook from top to bottom.

The notebook will:

1.  Load the processed dataset.
2.  Check data quality.
3.  Create text statistics.
4.  Visualize target and text distributions.
5.  Split the data into training and testing sets.
6.  Fit TF-IDF.
7.  Train Logistic Regression.
8.  Generate predictions.
9.  Calculate evaluation metrics.
10. Display the confusion matrix.
11. Display the ROC curve.
12. Analyze important TF-IDF terms.

------------------------------------------------------------------------

# 24. Business Interpretation

The model is intended to support recruitment analytics rather than
replace recruiter judgment.

Potential business applications include:

-   Identifying characteristics of job descriptions associated with
    stronger candidate interest.
-   Comparing language patterns across high- and low-performing
    listings.
-   Supporting data-driven job-description improvement.
-   Helping recruitment teams evaluate the wording and structure of
    future postings.
-   Identifying commonly occurring terms and phrases associated with the
    target class.

The model should be treated as a decision-support tool. Model
associations do not establish causal relationships.

------------------------------------------------------------------------

# 25. Limitations

Several limitations should be considered.

### Application-count censoring

LinkedIn may display applicant information in categories such as:

``` text
Over 200 applicants
Be among the first 25 applicants
```

These are not exact applicant counts.

### Platform-specific data

The observations come from LinkedIn job listings and therefore may not
represent every recruitment platform.

### Temporal variation

Job-market behavior changes over time. Patterns observed in one
collection period may not remain unchanged.

### Geographic variation

The dataset can contain listings from different locations, so candidate
behavior may vary by geography and labor market.

### Class-label definition

The high-performing label is constructed from application-count
information. The threshold therefore represents the project's analytical
definition of performance rather than an official LinkedIn
classification.

### Text-model interpretation

TF-IDF coefficients indicate associations in the training data. They
should not be interpreted as causal evidence.


------------------------------------------------------------------------

# 26. Ethical and Responsible Data Use

This project is intended for academic/business-analytics purposes.

The analysis focuses on job-listing attributes rather than attempting to
identify or profile individual applicants.

When using scraped data, users should:

-   Respect the terms and policies of the relevant platform.
-   Avoid exposing unnecessary personal information.
-   Avoid publishing private or sensitive information.
-   Use the data only for legitimate analytical purposes.
-   Clearly document how the data was collected and processed.

------------------------------------------------------------------------

# 27. Final Deliverables

The project consists of:

``` text
README.md
analysis.ipynb
data/dataset_processing.ipynb
data/datasetLinkedInraw.csv
data/datasetLinkedInprocessed.csv
```

Together, these files document the data pipeline, analytical
methodology, NLP model, and evaluation process.

------------------------------------------------------------------------

# 28. Project Summary

This project demonstrates an end-to-end Business Analytics workflow for
recruitment data:

``` text
Data Collection
      ↓
Data Cleaning
      ↓
Target Construction
      ↓
Exploratory Data Analysis
      ↓
Text Feature Engineering
      ↓
TF-IDF
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Interpretation
      ↓
Business Insights
```

The core analytical idea is to combine **recruitment data,
natural-language processing, and classification** to understand and
predict patterns in job-listing performance.

------------------------------------------------------------------------

## Author

**Anupama Nair**

**CB.SC.U4CSE23207**

B.Tech Computer Science and Engineering\
Amrita Vishwa Vidyapeetham, Coimbatore

**Project:** Business Analytics for Predicting High-Performing Job
Listings
