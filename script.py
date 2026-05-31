import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import normaltest

# ============================================================
# SECTION 1 - Load & Inspect Data
# ============================================================

df = pd.read_csv("carbon_emission_dataset_with_Industry.csv")

print(df.head())
print(df.info())
print(df.describe())

# ============================================================
# SECTION 2 - Data Cleaning
# ============================================================

print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

print(df.dtypes)

# ============================================================
# SECTION 3 - Exploratory Analysis
# ============================================================

cols = [
    "Total_Energy_Consumption_kWh",
    "Renewable_Energy_Consumption_kWh",
    "NonRenewable_Energy_Consumption_kWh",
    "Production_Output_Units",
    "Supply_Chain_Transport_km"
]

print(df[cols].agg(["mean", "median", "max", "min"]))

for col in cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in cols:
    df[col].value_counts().head(20).plot(kind="bar")
    plt.title(f"Frequency of {col}")
    plt.show()

df.set_index("Date")[["Total_Energy_Consumption_kWh",
                       "Renewable_Energy_Consumption_kWh",
                       "NonRenewable_Energy_Consumption_kWh"]].resample("ME").mean().plot()
plt.title("Energies Over Time")
plt.show()

plt.scatter(df["Energy_Cost_USD"], df["Carbon_Emission_tCO2e_TARGET"], alpha=0.3)
plt.xlabel("Energy Cost (USD)")
plt.ylabel("Carbon Emission (tCO2e)")
plt.title("Emissions vs Energy Cost")
plt.show()

plt.scatter(df["Date"], df["Carbon_Emission_tCO2e_TARGET"], alpha=0.3)
plt.title("Emissions vs Time")
plt.show()

# ============================================================
# SECTION 4 - Bivariate & Multivariate Analysis
# ============================================================

sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(data=df, x="Industry_Sectors", y="Carbon_Emission_tCO2e_TARGET")
plt.xticks(rotation=45)
plt.title("Emissions by Industry")
plt.show()

sns.pairplot(df[["Carbon_Emission_tCO2e_TARGET",
                  "Total_Energy_Consumption_kWh",
                  "Production_Output_Units"]].dropna())
plt.show()

print(df.groupby("Industry_Sectors")["Carbon_Emission_tCO2e_TARGET"].mean())

# ============================================================
# SECTION 5 - Time Series Exploration
# ============================================================

df.set_index("Date")["Carbon_Emission_tCO2e_TARGET"].resample("ME").mean().plot()
plt.title("Emissions Over Time")
plt.show()

df.set_index("Date")["Energy_Cost_USD"].resample("ME").mean().plot()
plt.title("Energy Cost Trend Over Time")
plt.show()

ts = df.set_index("Date")["Carbon_Emission_tCO2e_TARGET"].resample("ME").mean().dropna()
result = seasonal_decompose(ts, model="additive", period=6)
result.plot()
plt.show()

ts.rolling(3).mean().plot(label="Moving Average")
ts.rolling(3).var().plot(label="Rolling Variance")
plt.legend()
plt.title("Rolling Statistics")
plt.show()

# ============================================================
# SECTION 6 - Advanced Questions
# NOTE: We made the code to answer the questions to make sure we get it :)
# ============================================================

# Q1: Which emission sources have highest variability?
print("Q1 - Highest Variability:")
print(df.select_dtypes(include=np.number).std().sort_values(ascending=False).head(5))

# Q2: Are emissions normally distributed?
stat, p = normaltest(df["Carbon_Emission_tCO2e_TARGET"].dropna())
print(f"\nQ2 - Normal Distribution Test: p-value = {p:.5f}")
if p < 0.05:
    print("NOT normally distributed")
else:
    print("Normally distributed")

# Q3: Which variables are strongly correlated with emissions?
print("\nQ3 - Correlations with Emissions:")
corr = df.select_dtypes(include=np.number).corr()["Carbon_Emission_tCO2e_TARGET"]
print(corr.sort_values(ascending=False))

# Q4: Is there a trend in emissions?
slope = np.polyfit(range(len(ts)), ts.values, 1)[0]
print(f"\nQ4 - Trend slope: {slope:.4f}")
print("INCREASING" if slope > 0 else "DECREASING")

# Q5: Do seasonal patterns exist?
df["Month"] = df["Date"].dt.month
df.groupby("Month")["Carbon_Emission_tCO2e_TARGET"].mean().plot(kind="bar")
plt.title("Q5: Average Emissions by Month")
plt.xlabel("Month")
plt.show()