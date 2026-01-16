print("Script started")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_excel("data.xlsm")
# Rename columns
df = df.rename(columns={
    "Start time UTC": "datetime",
    "Electricity consumption (MWh)": "consumption"
})

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.dropna()

# Feature engineering
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

X = df[["hour", "day", "month", "year"]]
y = df["consumption"]

# Time-based split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, preds))
rmse = mean_squared_error(y_test, preds) ** 0.5
print("RMSE:", rmse)
