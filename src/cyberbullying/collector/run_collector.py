# run_collector.py

import random
import os
from dotenv import load_dotenv

from cyberbullying.collector.reddit_collector import fetch_all_reddit_content
from cyberbullying.collector.twitter_collector import fetch_all_twitter_content
from cyberbullying.collector.youtube_collector import fetch_all_youtube_content

from cyberbullying.collector.reddit_collector import send_to_api as send_reddit
from cyberbullying.collector.twitter_collector import send_to_api as send_twitter
from cyberbullying.collector.youtube_collector import send_to_api as send_youtube

load_dotenv()
# ---------------------------
# 🔹 GLOBAL DEDUP
# ---------------------------
seen_hashes = set()

def is_duplicate(text):
    key = text.strip().lower()

    if key in seen_hashes:
        return True

    seen_hashes.add(key)
    return False


# ---------------------------
# 🔹 FETCH ALL
# ---------------------------
def fetch_all_platforms():

    data = []

    data.extend(fetch_all_reddit_content())
    if os.getenv("ENABLE_TWITTER") == "true":
        data.extend(fetch_all_twitter_content())
    data.extend(fetch_all_youtube_content())

    random.shuffle(data)

    return data

# def fetch_all_platforms():

#     data = []

#     data.extend(fetch_all_reddit_content())

#     # 🔹 Twitter (SAFE MODE)
#     try:
#         twitter_data = fetch_all_twitter_content()
#         if twitter_data:
#             data.extend(twitter_data)
#     except Exception as e:
#         print("Twitter skipped:", e)

#     data.extend(fetch_all_youtube_content())

#     random.shuffle(data)

#     return data


# ---------------------------
# 🔹 SEND ROUTER
# ---------------------------
def send_item(item):

    if item["platform"] == "reddit":
        return send_reddit(item)

    elif item["platform"] == "twitter":
        return send_twitter(item)

    elif item["platform"] == "youtube":
        return send_youtube(item)

    return {"error": "Unknown platform"}


# ---------------------------
# 🔹 SINGLE RUN FUNCTION
# ---------------------------
def run_once():

    results = []

    data = fetch_all_platforms()

    # ✅ LIMIT TOTAL FETCHED
    data = data[:40]
    
    MAX_ITEMS = 15   # or 20 

    for item in data:

        if len(results) >= MAX_ITEMS:
            break

        if is_duplicate(item["text"]):
            continue

        result = send_item(item)

        results.append({
            "text": item["text"][:100],
            "platform": item["platform"],
            "content_type": item["content_type"],
            "result": result
        })

    return {
        "total_fetched": len(data),
        "processed": len(results),
        "results": results[:10]   # limit preview
    }