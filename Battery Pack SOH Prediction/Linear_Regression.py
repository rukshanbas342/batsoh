# Importing libraries
import pandas as pd
import numpy as np
import time  #for getting execution time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

import os # To save the plot inside Results folder

# Create folder if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

start = time.time()

# =====================================
# Data Preprocessing and Cleaning
# =====================================

# Loading dataset form the folder Data
data = pd.read_excel("Data/PulseBat Dataset.xlsx")

# Checking the column headers in excel file -- for testing purposes 
print("Columns in dataset:", list(data.columns))

# Select only U1–U21 and SOH since they are of interest
columns_to_use = [f"U{i}" for i in range(1, 22)] + ["SOH"]
data = data[columns_to_use]

# Making sure that correct columns are selected using the above algorithm
print("\n✅ Selected columns:")
print(data.head())  # column headers

# Handle missing values
before = data.shape[0]
data = data.dropna()  # drops row with any missing value
after = data.shape[0]
print(f"\nDropped {before - after} rows with missing values. Final shape: {data.shape}")

# Saved cleaned data to a new excel file
data.to_excel("Data/Cleaned_PulseBat_Dataset.xlsx", index=False) # just to make sure what our cleaned dtaa set looks like
print("Cleaned dataset saved as 'Cleaned_PulseBat_Dataset.xlsx'")


# Sorting techniques
# We will sort these 21 values in 3 different orders for testing
# Extracts voltage values from U1 - U21 for sorting

X = data.loc[:, "U1":"U21"].to_numpy() #directly to NumPy, faster than using .values later
y = data["SOH"]  # Extracts SOH column from dataframe

# Create 3 orders
unsorted_data = X.copy()
ascending_data = np.sort(X, axis=1) # sorts in ascending using numpy function
descending_data = ascending_data[:, ::-1]  # FLipping the ascending array


# Converting to DataFrames using Pandas function
ascending_df = pd.DataFrame(ascending_data, columns=[f"U{i}" for i in range(1, 22)]) 
descending_df = pd.DataFrame(descending_data, columns=[f"U{i}" for i in range(1, 22)])

# Verify sorting visually -for testing purposes
print("\nOriginal first row:")
print(X[0])

print("\nAscending sorted row:")
print(ascending_df.iloc[0].values)

print("\nDescending sorted row:")
print(descending_df.iloc[0].values)

# Creating a dictionary to store all the versions
datasets = {
    "Unsorted": unsorted_data,
    "Ascending": ascending_data,
    "Descending": descending_data
}

# =====================================
# Linear Regression Model & Evaluation
# =====================================

# Loops through each item in the dataset disctionary
for name, X_data in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y, test_size=0.2, random_state=42 # 80% training, 20% testing and random_state=42 ensures reproducibility
        )
    
    # Training the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting on test set
    y_pred = model.predict(X_test)

    # Evaluating model performance
    r2 = r2_score(y_test, y_pred) # R^2 --> How well the model explains variance
    mse = mean_squared_error(y_test, y_pred) 
    mae = mean_absolute_error(y_test, y_pred) 

    # Printing the Analytics
    print(f"\n📊 {name} Data Model Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

end = time.time()
print(f"\n⏱️ Total script execution time: {end - start:.4f} seconds")

# ================================
# Battery Classification Section
# ================================

# user Input for threshold (default = 0.6) SOH >= threshold is considered healthy else unhealthy
threshold = float(input("\nEnter SOH threshold for classification (e.g., 0.6): ") or 0.6)

# Classify based on threshold both for predicted SOH and Actuacl SOH
y_pred_class = np.where(y_pred >= threshold, "Healthy", "Unhealthy")
y_test_class = np.where(y_test >= threshold, "Healthy", "Unhealthy")

# Combine results into a DataFrame for later viewing
results_df = pd.DataFrame({
    "Actual SOH": y_test,   
    "Predicted SOH": y_pred,
    "Actual Class": y_test_class,
    "Predicted Class": y_pred_class
})

# ====================================
# Plotting and Visualization Section 
# ====================================

# Scatter plot --> Actual SOH vs Predicted SOH
plt.figure(figsize=(8,6))

# Determine colors based on threshold
# Green → Predicted as Healthy, Red → Predicted as Unhealthy
colors = np.where(y_pred >= threshold, 'green', 'red')

plt.scatter(y_test, y_pred, color=colors, edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # helps see how far predictions are from actual values
plt.axhline(y=threshold, color='orange', linestyle='--', lw=2) # Threshold line

plt.gca().text(0.05, 0.95, f'Threshold = {threshold}', transform=plt.gca().transAxes,  # Threshold text at top-left
               fontsize=12, fontweight='bold', verticalalignment='top', color='orange',
               bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

# Count of Healthy and Unhealthy predictions
healthy_count = np.sum(y_pred >= threshold)
unhealthy_count = np.sum(y_pred < threshold)

# Healthy/Unhealthy count text at bottom-right
plt.gca().text(0.95, 0.05, f"Healthy: {healthy_count}\nUnhealthy: {unhealthy_count}",
               transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='right',
               color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

# Labels, title, grid
plt.title("Actual SOH vs Predicted SOH", fontsize=14, fontweight='bold')
plt.xlabel("Actual SOH", fontsize=10)
plt.ylabel("Predicted SOH", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("results/Actual_vs_Predicted_SOH.png", dpi=300) # plot saved to results folder
plt.show()

# Residuals/Errors Plot
# A residual tells you how much your model over- or under-predicted for each data point
residuals = y_test - y_pred 

plt.figure(figsize=(8,5))

# Plot histogram to visualize the distribution of residuals/errors
sns.histplot(residuals, bins=25, kde=True, color='skyblue', edgecolor='black', alpha=0.7)

# Vertical line at 0 for perfect predictions
plt.axvline(0, color='black', linestyle='--', lw=2, label='Zero Error')

# Plot title and axis labels
plt.title("Distribution of Residual Errors", fontsize=14, fontweight='bold')
plt.xlabel("Residual (Actual - Predicted) / Error", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Counting for extreme errors: negative and positive both
large_error_count = np.sum(np.abs(residuals) > 0.05)  # example threshold
plt.text(0.95, 0.95, f'|Residual| > 0.05: {large_error_count}', 
         transform=plt.gca().transAxes, ha='right', va='top', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

plt.tight_layout()
plt.savefig("results/Residuals_Distribution.png", dpi=300) # plot saved to results folder
plt.show()

# Create a copy with new index starting from 1
results_sample = results_df.head(10).copy()
results_sample.index = range(1, len(results_sample)+1)

print("\n🔋 Sample predictions with classification:\n")
print(results_sample)