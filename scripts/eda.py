import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data/revenue.csv", parse_dates=["date"])
    os.makedirs("artifacts", exist_ok=True)
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    plt.figure(figsize=(10, 4))
    plt.plot(daily["date"], daily["revenue"])
    plt.title("Total Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig("artifacts/daily_revenue.png")
    plt.close()
    country = df.groupby("country")["revenue"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    country.plot(kind="bar")
    plt.title("Average Revenue by Country")
    plt.xlabel("Country")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.savefig("artifacts/revenue_by_country.png")
    plt.close()
    print("EDA charts saved in artifacts/")

if __name__ == "__main__":
    main()
