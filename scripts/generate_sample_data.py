import os
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    countries = ["United Kingdom", "Canada", "Germany", "Australia"]
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    rows = []
    country_base = {"United Kingdom": 1400, "Canada": 1100, "Germany": 1250, "Australia": 1000}
    for country in countries:
        for d in dates:
            seasonal = 220 * np.sin(2 * np.pi * d.dayofyear / 365.25)
            weekend_effect = -100 if d.weekday() >= 5 else 80
            trend = (d.year - 2023) * 65
            noise = rng.normal(0, 120)
            revenue = max(50, country_base[country] + seasonal + weekend_effect + trend + noise)
            rows.append({"date": d.strftime("%Y-%m-%d"), "country": country, "revenue": round(float(revenue), 2)})
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(rows).to_csv("data/revenue.csv", index=False)
    print(f"Wrote data/revenue.csv ({len(rows)} rows; synthetic demo data).")

if __name__ == "__main__":
    main()
