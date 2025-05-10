import os
import re
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


DATA_DIR = r"C:\\Users\\ENJD\\Desktop\\Bluesky"
OUTPUT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_FILE_XLSX = f"FUTURE_{OUTPUT_TIME}.xlsx"
OUTPUT_FILE_CSV = f"FUTURE_{OUTPUT_TIME}.csv"

# DATE
def extract_date_from_filename(filename):
    match = re.search(r'post(\d{8})\d{4}', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return None

# ENGEGEMENT ALL
def aggregate_daily_engagement():
    data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".xlsx") and file.startswith("post"):
            filepath = os.path.join(DATA_DIR, file)
            file_date = extract_date_from_filename(file)
            if file_date:
                try:
                    df = pd.read_excel(filepath, engine="openpyxl")
                    daily_total = df[["Post Like Count", "Repost Count", "Comment Count"]].sum().sum()
                    data.append({"date": file_date, "engagement": daily_total})
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    return pd.DataFrame(data).sort_values("date")

# TRAIN
def predict_next_day(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])  
    df["day"] = (df["date"] - df["date"].min()).dt.days

    X = df[["day"]]
    y = df["engagement"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    tomorrow_day = df["day"].max() + 1
    pred = model.predict([[tomorrow_day]])[0]
    pred_date = df["date"].max() + timedelta(days=1)

    print(f"Predicted engagement for {pred_date}: {int(pred)}")

    # V
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["engagement"], marker='o', label="Historical")
    plt.axvline(pred_date, color='red', linestyle='--', label="Prediction Date")
    plt.scatter([pred_date], [pred], color='orange', label="Predicted")
    plt.title("Daily Engagement & Prediction")
    plt.xlabel("Date")
    plt.ylabel("Engagement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    result_df = pd.DataFrame([{
        "Predict_Date": pred_date.strftime("%Y-%m-%d"),
        "Predicted_Engagement": int(pred),
        "Model_Run_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    result_df.to_excel(OUTPUT_FILE_XLSX, index=False)
    result_df.to_csv(OUTPUT_FILE_CSV, index=False)
    print(f"✅ : {OUTPUT_FILE_XLSX}, {OUTPUT_FILE_CSV}")


if __name__ == "__main__":
    df = aggregate_daily_engagement()
    if len(df) < 5:
        print("MORE THAN 5")
    else:
        predict_next_day(df)