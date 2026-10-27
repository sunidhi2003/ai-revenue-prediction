from datetime import date

def make_features(target_date: str, country: str, countries: list[str]) -> dict:
    d = date.fromisoformat(target_date)
    return {
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "day_of_week": d.weekday(),
        "day_of_year": d.timetuple().tm_yday,
        "is_weekend": int(d.weekday() >= 5),
        "country": country,
    }
