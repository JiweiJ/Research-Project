import os
import re
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


DATA_DIR = r"C:\\Users\\ENJD\\Desktop\\Bluesky"
OUTPUT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_FILE_XLSX = f"KEYWORD_PREDICT_{OUTPUT_TIME}.xlsx"
OUTPUT_FILE_CSV = f"KEYWORD_PREDICT_{OUTPUT_TIME}.csv"

# DATE
def extract_date_from_filename(filename):
    match = re.search(r'post(\d{8})\d{4}', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return None


def load_all_data():
    records = []
    for file in os.listdir(DATA_DIR):
        if file.startswith("post") and file.endswith(".xlsx"):
            path = os.path.join(DATA_DIR, file)
            date = extract_date_from_filename(file)
            try:
                df = pd.read_excel(path, engine="openpyxl")
                df = df[["Query", "Post Like Count", "Repost Count", "Comment Count"]].copy()
                df = df.dropna(subset=["Query"])
                df["Engagement"] = df[["Post Like Count", "Repost Count", "Comment Count"]].sum(axis=1)
                df["Date"] = date
                records.append(df)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# TRAIN
def train_and_predict(df):
    results = []
    grouped = df.groupby("Query")
    predict_date = df["Date"].max() + timedelta(days=1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for query, group in grouped:
        daily_stats = group.groupby("Date").agg(Post_Count=("Engagement", "count"),
                                                  Total_Engagement=("Engagement", "sum"))
        daily_stats = daily_stats[daily_stats["Post_Count"] >= 20].copy()
        if len(daily_stats) < 5:
            continue

        daily_stats = daily_stats.reset_index()
        daily_stats["Date"] = pd.to_datetime(daily_stats["Date"]) 
        daily_stats["day"] = (daily_stats["Date"] - daily_stats["Date"].min()).dt.days

        X = daily_stats[["day"]]
        y = daily_stats["Total_Engagement"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = lgb.LGBMRegressor()
        model.fit(X_train, y_train)

        next_day = daily_stats["day"].max() + 1
        pred = model.predict([[next_day]])[0]

        results.append({
            "Query": query,
            "Predict Date": predict_date.strftime("%Y-%m-%d"),
            "Predicted Engagement": int(pred),
            "Last N Days Used": len(daily_stats),
            "Model Time": now
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df_all = load_all_data()
    if df_all.empty:
        print("❌ No data available.")
    else:
        result_df = train_and_predict(df_all)
        result_df.to_excel(OUTPUT_FILE_XLSX, index=False)
        result_df.to_csv(OUTPUT_FILE_CSV, index=False)
        print(f"✅  {OUTPUT_FILE_XLSX} and {OUTPUT_FILE_CSV}")