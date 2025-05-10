import requests
import csv
import json
import time
import random
import pandas as pd
import concurrent.futures
import threading
import os
from datetime import datetime

USERNAME = "jw-j.bsky.social"
APP_PASSWORD = "ni4b-fkw2-fmf6-xigc"
BASE_URL = "https://bsky.social/xrpc"
TEST_MODE = False
TEST_QUERY_LIMIT = 2


# Get token
def get_token(username, app_password):
    url = f"{BASE_URL}/com.atproto.server.createSession"
    payload = {
        "identifier": username,
        "password": app_password
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        token = data.get("accessJwt")
        if token:
            print("✅ Successfully obtained token.")
            return token
    print("❌ Failed to get token.")
    return None

# Search posts
def search_posts(query, token, limit=200):
    url = f"{BASE_URL}/app.bsky.feed.searchPosts"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "limit": 50}
    posts = []
    cursor = None
    while len(posts) < limit:
        if cursor:
            params["cursor"] = cursor
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Search error: {query}")
            break
        data = response.json()
        batch = data.get("posts", [])
        if not batch:
            break
        posts.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(random.uniform(0.05, 0.2))
    return posts[:limit]

# Get actor info
def get_actor_info(handle, token):
    url = f"{BASE_URL}/app.bsky.actor.getProfile"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"actor": handle}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            "followersCount": data.get("followersCount", 0),
            "postsCount": data.get("postsCount", 0)
        }
    return {"followersCount": None, "postsCount": None}

# Get comment likes
def get_comment_likes(comment_uri, token, limit=100):
    url = f"{BASE_URL}/app.bsky.feed.getLikes"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"uri": comment_uri, "limit": limit}
    response = requests.get(url, headers=headers, params=params)
    time.sleep(random.uniform(1, 2))
    if response.status_code == 200:
        data = response.json()
        like_count = data.get("count")
        if like_count is None:
            like_count = len(data.get("likes", []))
        return like_count
    return 0

# Get comments
def get_comments(post_uri, token, limit=200):
    url = f"{BASE_URL}/app.bsky.feed.getPostThread"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"uri": post_uri, "limit": limit}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"❌ Failed to get comments for {post_uri}")
        return []
    data = response.json()
    thread = data.get("thread", {})
    all_comments = []

    def collect_replies(node):
        if isinstance(node, dict):
            for reply in node.get("replies", []):
                post_data = reply.get("post", {})
                record = post_data.get("record", {})
                author = post_data.get("author", {})
                comment_uri = post_data.get("uri", "")
                like_count = 0
                if comment_uri:
                    like_count = get_comment_likes(comment_uri, token)
                comment_item = {
                    "text": record.get("text", ""),
                    "createdAt": record.get("createdAt", ""),
                    "author_handle": author.get("handle", ""),
                    "author_displayName": author.get("displayName", ""),
                    "likeCount": like_count
                }
                all_comments.append(comment_item)
                collect_replies(reply)

    collect_replies(thread)
    return all_comments[:limit]

# Process posts
def process_posts(posts, token, author_cache, cache_lock):
    processed = []
    for post in posts:
        record = post.get("record", {})
        content = record.get("text", "")
        created_at = record.get("createdAt", "")
        author_info = post.get("author", {})
        author_handle = author_info.get("handle", "")
        like_count = post.get("likeCount", 0)
        repost_count = post.get("repostCount", 0)
        reply_count = post.get("replyCount", 0)
        tags = record.get("tags", [])
        if isinstance(tags, list):
            tags = ",".join(tags)
        with cache_lock:
            if author_handle in author_cache:
                actor_data = author_cache[author_handle]
            else:
                actor_data = get_actor_info(author_handle, token)
                author_cache[author_handle] = actor_data
        processed.append({
            "Post Content": content,
            "Post Time": created_at,
            "Author": author_handle,
            "Post Like Count": like_count,
            "Repost Count": repost_count,
            "Comment Count": reply_count,
            "Tags": tags,
            "Author Followers": actor_data.get("followersCount"),
            "Author Post Count": actor_data.get("postsCount"),
            "Post ID": post.get("uri", ""),
            "post_uri": post.get("uri", "")  
        })
    return processed

def process_comments(comments, source_post_uri):
    processed = []
    for comment in comments:
        processed.append({
            "Source Post": source_post_uri,
            "Comment Content": comment.get("text", ""),
            "Comment Like Count": comment.get("likeCount", 0),
            "Comment Time": comment.get("createdAt", ""),
            "Comment Author": comment.get("author_handle", "")
        })
    return processed

def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(filename, data, headers):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def write_excel(filename, data, headers):
    df = pd.DataFrame(data, columns=headers)
    df.to_excel(filename, index=False)

def process_query(query, token, author_cache, cache_lock):

    print(f"🔍 Starting query: {query}")
    posts = search_posts(query, token, limit=200)
    processed_posts = process_posts(posts, token, author_cache, cache_lock)
    for p in processed_posts:
        p["Query"] = query

    local_comments = []
    for post in processed_posts:
        post_uri = post.get("post_uri")
        if post_uri:
            try:
                comments = get_comments(post_uri, token)
                local_comments.extend(process_comments(comments, post_uri))
            except Exception as e:
                print(f"❌ Failed to fetch comments for {post_uri}: {str(e)}")
        else:
            print(f"⚠️ Skipping post without URI: {post.get('Post Content')[:30]}...")

    # with cache_lock:
    #     finished.add(query)
    #     with open(finished_queries_file, "w", encoding="utf-8") as f:
    #         json.dump(list(finished), f, ensure_ascii=False, indent=2)

    return processed_posts, local_comments



def main():
    token = get_token(USERNAME, APP_PASSWORD)
    if not token:
        return

    author_cache = {}
    cache_lock = threading.Lock()

    en_topic = ["Belt and Road Initiative", "One Belt One Road", "Maritime Silk Road",
                "Silk Road Economic Belt", "Chinese investment", "Chinese infrastructure projects",
                "Debt diplomacy", "Economic cooperation", "Strategic partnerships"]
    en_location = ["Papua New Guinea", "Solomon Islands", "Vanuatu", "Fiji"]
    zh_topic_simplified = ["一带一路倡议", "一带一路", "海上丝绸之路", "丝绸之路经济带",
                           "中国投资", "中国基础设施项目", "债务外交", "经济合作", "战略伙伴关系"]
    zh_location_simplified = ["巴布亚新几内亚", "所罗门群岛", "瓦努阿图", "斐济"]
    zh_topic_traditional = ["一帶一路倡議", "一帶一路", "海上絲路", "絲路經濟帶",
                            "中國投資", "中國基建項目", "債務外交", "經濟合作", "戰略夥伴關係"]
    zh_location_traditional = ["巴布亞紐幾內亞", "所羅門群島", "瓦努阿圖", "斐濟"]

    queries = en_topic + zh_topic_simplified + zh_topic_traditional
    queries += en_location + zh_location_simplified + zh_location_traditional
    queries += [f"{t} {l}" for t in en_topic for l in en_location]
    queries += [f"{t} {l}" for t in zh_topic_simplified for l in zh_location_simplified]
    queries += [f"{t} {l}" for t in zh_topic_traditional for l in zh_location_traditional]


    # default_finished_file = r"C:\Users\ENJD\Desktop\Bluesky\BSDR\finished_queries_latest.json"

    # if os.path.exists(default_finished_file):
    #     with open(default_finished_file, "r", encoding="utf-8") as f:
    #         finished = set(json.load(f))
    # else:
    #     finished = set()


    if TEST_MODE:
        queries = queries[:TEST_QUERY_LIMIT]
    # queries = [q for q in queries if q not in finished]


    output_dir = "/home/ubuntu/Bluesky"
    os.makedirs(output_dir, exist_ok=True)
    
    now_str = datetime.now().strftime("%Y%m%d%H%M")

    # finished_queries_file = os.path.join(output_dir, f"finished_queries{now_str}.json")

    all_posts, all_comments = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_query, q, token, author_cache, cache_lock) for q in queries]

        for future in concurrent.futures.as_completed(futures):
            posts, comments = future.result()
            all_posts.extend(posts)
            all_comments.extend(comments)
   
    post_csv_headers = ["Post Content", "Post Time", "Author", "Post Like Count", "Repost Count", "Comment Count",
                        "Tags", "Author Followers", "Author Post Count", "Post ID", "Query", "post_uri"]

    
    posts_csv_data = [{k: v for k, v in row.items() if k in post_csv_headers} for row in all_posts]

    write_json(os.path.join(output_dir, f"post{now_str}.json"), all_posts)
    write_csv(os.path.join(output_dir, f"post{now_str}.csv"), posts_csv_data, post_csv_headers)
    write_excel(os.path.join(output_dir, f"post{now_str}.xlsx"), all_posts, post_csv_headers)

    comment_csv_headers = ["Source Post", "Comment Content", "Comment Like Count", "Comment Time", "Comment Author"]

    write_json(os.path.join(output_dir, f"comment{now_str}.json"), all_comments)
    write_csv(os.path.join(output_dir, f"comment{now_str}.csv"), all_comments, comment_csv_headers)
    write_excel(os.path.join(output_dir, f"comment{now_str}.xlsx"), all_comments, comment_csv_headers)



    print("✅ Finished all queries. Data saved.")

if __name__ == "__main__":
    main()
