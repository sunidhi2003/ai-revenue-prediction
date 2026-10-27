import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.features import make_features

def main():
    data_path = "data/revenue.csv"
    if not os.path.exists(data_path):
        raise SystemExit("Missing data/revenue.csv. Run python scripts/generate_sample_data.py first.")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.dropna(subset=["date", "country", "revenue"])
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["day"] = df.date.dt.day
    df["day_of_week"] = df.date.dt.weekday
    df["day_of_year"] = df.date.dt.dayofyear
    df["is_weekend"] = (df.date.dt.weekday >= 5).astype(int)
    features = ["year", "month", "day", "day_of_week", "day_of_year", "is_weekend", "country"]
    X, y = df[features], df["revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    prep = ColumnTransformer([
        ("country", OneHotEncoder(handle_unknown="ignore"), ["country"]),
        ("numeric", StandardScaler(), features[:-1]),
    ])
    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": Pipeline([("prep", prep), ("model", LinearRegression())]),
        "random_forest": Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=3, n_jobs=-1))]),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
            "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 2),
        }
    best_name = min(("linear_regression", "random_forest"), key=lambda n: scores[n]["MAE"])
    best_model = models[best_name]
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/revenue_model.joblib")
    countries = sorted(df["country"].unique().tolist())
    joblib.dump(countries, "models/countries.joblib")
    os.makedirs("artifacts", exist_ok=True)
    pd.DataFrame(scores).T.to_csv("artifacts/model_comparison.csv")
    print("Model comparison (lower is better):")
    for name, score in scores.items():
        print(f"{name}: MAE={score['MAE']} RMSE={score['RMSE']}")
    print(f"Saved best model: {best_name}")

if __name__ == "__main__":
    main()
