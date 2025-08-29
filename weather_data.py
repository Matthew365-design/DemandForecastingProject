import requests
import pandas as pd

def get_weather(lat, lon, start, end, vars):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(vars),
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["Datetime"] = pd.to_datetime(df["time"])
    return df.set_index("Datetime")
