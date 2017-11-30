# Add the facets overview python code to the python path
import sys
# FACETS_PATH is the full path to the python file in the clonde github repo of Facets.
# It should look similar to this: ".../facets/facets_overview/python"
# If you have cloned the facets repo to your current working directory, you can proceed.
# If you have chosen another location, just add it here.

sys.path.append('./facets/facets_overview/python')
# Load UCI census train and test data into dataframes. 
# Pandas can directly read URL, download the files in the URLs below
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
import pandas as pd
features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"]
train_data = pd.read_csv(
    "./data/adult.data",
    names=features,
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
test_data = pd.read_csv(
    "./data/adult.test",
    names=features,
    sep=r'\s*,\s*',
    skiprows=[0],
    engine='python',
    na_values="?")