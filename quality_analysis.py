import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pycaret.classification import setup, compare_models, create_model, predict_model

# Load the dataset
data = pd.read_csv("water_potability.csv")
print(data.head())

print('\n')
# Check for null values before cleaning
null_val_before = data.isnull().sum()
print("Null values in dataset before data cleaning:\n", null_val_before)

print('\n')
# Drop rows with null values
data.dropna(inplace=True)
null_val_after = data.isnull().sum()
print("Null values in dataset after data cleaning:\n", null_val_after)

# Plot the distribution of water potability
plt.figure(figsize=(15, 10))
sns.countplot(data=data, x="Potability")
plt.title("Distribution of Safe and Unsafe Water")
plt.show()

# Plotly histograms for different features affecting water quality
figure = px.histogram(data, x="ph", color="Potability", title="Factors Affecting Water Quality: ph")
figure.show()

figure = px.histogram(data, x="Hardness", color="Potability", title="Factors Affecting Water Quality: Hardness")
figure.show()

figure = px.histogram(data, x="Solids", color="Potability", title="Factors Affecting Water Quality: Solids")
figure.show()

figure = px.histogram(data, x="Chloramines", color="Potability", title="Factors Affecting Water Quality: Chloramines")
figure.show()

figure = px.histogram(data, x="Sulfate", color="Potability", title="Factors Affecting Water Quality: Sulfate")
figure.show()

figure = px.histogram(data, x="Conductivity", color="Potability", title="Factors Affecting Water Quality: Conductivity")
figure.show()

figure = px.histogram(data, x="Organic_carbon", color="Potability",
                      title="Factors Affecting Water Quality: Organic_carbon")
figure.show()

figure = px.histogram(data, x="Trihalomethanes", color="Potability",
                      title="Factors Affecting Water Quality: Trihalomethanes")
figure.show()

figure = px.histogram(data, x="Turbidity", color="Potability", title="Factors Affecting Water Quality: Turbidity")
figure.show()


# Correlation matrix
correlation = data.corr()
print(correlation["ph"].sort_values(ascending=False))

# Set up the classification model with PyCaret
clf = setup(data, target="Potability", session_id=786)

# Compare models
best_model = compare_models()

# Create and evaluate a random forest model
model = create_model("rf")
predict = predict_model(model, data=data)
print(predict.head())

