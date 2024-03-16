# Forecasting-CoCo-CoLa-prices
Forecast the CocaCola prices. Prepare a document for each model explaining  how many dummy variables you have created and RMSE value for each model. Finally which model you will use for  Forecasting.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset from Excel file
data = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')

# Extract year and quarter from the 'Quarter' column
data[['Quarter', 'Year']] = data['Quarter'].str.split('_', expand=True)

# Combine year and quarter to form a new date column
data['Date'] = pd.to_datetime(data['Year'] + '-' + data['Quarter'].str.replace('Q', '') + '-01')

# Set the new date column as the index
data.set_index('Date', inplace=True)

# Drop the original 'Year' and 'Quarter' columns
data.drop(['Year', 'Quarter'], axis=1, inplace=True)

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Sales'], marker='o', linestyle='-')
plt.title('Coca-Cola Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Sales'], marker='o', linestyle='-')
plt.title('Coca-Cola Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform decomposition
decomposition = seasonal_decompose(data['Sales'], model='additive', period=4)

# Plot the decomposition
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.title('Decomposition Plot of Coca-Cola Sales')
plt.show()



# Autocorrelation Plot
plt.figure(figsize=(10, 6))
pd.plotting.autocorrelation_plot(data['Sales'])
plt.title('Autocorrelation Plot of Coca-Cola Sales')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], kde=True, color='green', bins=20)
plt.title('Distribution of Coca-Cola Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.index, y=data['Sales'], color='red')
plt.title('Coca-Cola Sales Over Time (Scatter Plot)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Pairplot for Multivariate Analysis
sns.pairplot(data=data, diag_kind='kde')
plt.title('Pairplot for Multivariate Analysis')
plt.show()

# SARIMA Model Selection
best_rmse = float('inf')
best_model = None

for p in range(0, 3):
    for d in range(1, 3):
        for q in range(0, 3):
            for P in range(0, 3):
                for D in range(0, 2):
                    for Q in range(0, 3):
                        try:
                            model = SARIMAX(data['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, 4))
                            fitted_model = model.fit(disp=False)
                            forecast = fitted_model.forecast(steps=12)
                            rmse = mean_squared_error(data['Sales'][-12:], forecast, squared=False)

                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_model = fitted_model

                            print(f"SARIMA({p}, {d}, {q})({P}, {D}, {Q}, 4) - RMSE: {rmse}")
                        except:
                            continue

# Final model for forecasting
if best_model is not None:
    print("Best Model:")
    print(best_model.summary())

    # Forecast using the best model
    forecast = best_model.forecast(steps=12)

    # Calculate RMSE for the best model
    rmse = mean_squared_error(data['Sales'][-12:], forecast, squared=False)
    print("RMSE for the Best Model:", rmse)
else:
    print("No valid model found.")
