import os
import json
import re
import requests
from datetime import datetime, timedelta

BASE_DIR = "/home/ubuntu/Bluesky"

SLACK_WEBHOOK = "This is my slack url"

FILENAME_PATTERN = re.compile(r"post(\d{8})\d{4}\.json")

def find_post_file_by_date(target_date: datetime):
    date_str = target_date.strftime("%Y%m%d")
    for filename in os.listdir(BASE_DIR):
        if FILENAME_PATTERN.match(filename) and filename.startswith(f"post{date_str}"):
            return os.path.join(BASE_DIR, filename)
    return None

def load_post_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return {post["Post ID"]: post for post in json.load(f)}

def compare_posts(today_data, yesterday_data):
    new_posts = []
    changed_posts = []
    unchanged_posts = []

    for post_id, today_post in today_data.items():
        if post_id not in yesterday_data:
            new_posts.append(today_post)
        else:
            y_post = yesterday_data[post_id]
            if (today_post.get("Post Like Count") != y_post.get("Post Like Count") or
                today_post.get("Repost Count") != y_post.get("Repost Count") or
                today_post.get("Comment Count") != y_post.get("Comment Count")):
                changed_posts.append(today_post)
            else:
                unchanged_posts.append(today_post)

    return {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Newly added posts today": len(new_posts),
        "Active posts today": len(changed_posts),
        "Posts no activity": len(unchanged_posts),
        "Final tracked posts": len(today_data)
    }

def send_to_slack(text):
    payload = {"text": text}
    try:
        response = requests.post(SLACK_WEBHOOK, json=payload)
        if response.status_code == 200:
            print("✅ ")
        else:
            print(f"❌ Slack  {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Slack : {e}")


def main():
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    today_file = find_post_file_by_date(today)
    yesterday_file = find_post_file_by_date(yesterday)

    if not today_file or not yesterday_file:
        print("❌ No post file")
        print(f"✔️ File：{today_file}")
        print(f"✔️ Yfile：{yesterday_file}")
        return

    today_data = load_post_data(today_file)
    yesterday_data = load_post_data(yesterday_file)
    summary = compare_posts(today_data, yesterday_data)

    summary_text = (
        f"\n📘 *Bluesky Daily Summary*\n"
        f"Date: {summary['Date']}\n"
        f"Newly added posts today: {summary['Newly added posts today']}\n"
        f"Active posts today: {summary['Active posts today']}\n"
        f"Posts no activity: {summary['Posts no activity']}\n"
        f"Final tracked posts: {summary['Final tracked posts']}"
    )

    print(summary_text)
    send_to_slack(summary_text)  

if __name__ == "__main__":
    main()
