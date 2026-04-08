# run_collector.py
import time
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

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
# ---------------------------
# 🔹 GLOBAL DEDUP (changed to use ID)
# ---------------------------
seen_ids = set()

def is_duplicate(item):
    # Try to get the real ID, fallback to text if the ID is missing
    post_id = item.get("platform_post_id") or item.get("text", "").strip().lower()

    if post_id in seen_ids:
        return True

    seen_ids.add(post_id)
    return False


# ---------------------------
# 🔹 FETCH ALL
# ---------------------------
def fetch_all_platforms():

    data = []

  
    
    if DEMO_MODE:
        # 🔥 MODE A: The "Forced" Demo Mode
        # We skip the natural feeds and ONLY pull targeted languages 
        # to guarantee a perfectly diverse dashboard for the examiner.
        print("🎯 DEMO MODE ACTIVE: Forcing multilingual collection...")
        
        # Directly call the targeted functions from the platform files
        from cyberbullying.collector.reddit_collector import fetch_targeted_reddit
        from cyberbullying.collector.twitter_collector import fetch_targeted_tweets
        from cyberbullying.collector.youtube_collector import fetch_targeted_youtube
        
        target_langs = ["hindi", "marathi", "tamil", "bengali", "gujarati"]
        for lang in target_langs:
            data.extend(fetch_targeted_reddit(lang, limit=2))
            data.extend(fetch_targeted_youtube(lang))
            if os.getenv("ENABLE_TWITTER") == "true":
                try:
                    data.extend(fetch_targeted_tweets(lang, limit=2))
                except:
                    pass
    else:
        # 🌿 MODE B: The "Natural + Balancing" Mode
        # Pulls natural feeds, and uses his balancing logic to fill gaps.
        print("🌿 NATURAL MODE ACTIVE: Collecting trending feeds with dynamic balancing...")
        
        data.extend(fetch_all_reddit_content())
        if os.getenv("ENABLE_TWITTER") == "true":
            try:
                data.extend(fetch_all_twitter_content())
            except Exception as e:
                print("Twitter skipped:", e)
        data.extend(fetch_all_youtube_content())

    # Shuffle the data so the dashboard doesn't show all Reddit first, then all YouTube
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
# 🔹 SINGLE RUN FUNCTION (Advanced Logging added)
# ---------------------------
def run_once():
    results = []
    
    print("\n" + "="*50)
    print(f"🚀 STARTING DATA COLLECTOR | DEMO_MODE: {DEMO_MODE}")
    print("="*50)

    # ⏱️ 1. TRACK FETCH TIME
    start_fetch = time.time()
    data = fetch_all_platforms()
    fetch_ms = round((time.time() - start_fetch) * 1000, 2)
    
    print(f"📥 Fetched {len(data)} items across platforms in {fetch_ms} ms")
    print("-" * 50)

    # ⏱️ 2. TRACK PROCESSING TIME
    start_process = time.time()
    
    # ✅ LIMIT TOTAL PROCESSED
    MAX_ITEMS = 15  
    processed_count = 0

    for item in data:
        if processed_count >= MAX_ITEMS:
            break

        # 🔥 Duplicate check using the new ID logic
        if is_duplicate(item):
            continue

        result = send_item(item)

        results.append({
            "text": item["text"][:100],
            "platform": item["platform"],
            "content_type": item["content_type"],
            "result": result
        })
        processed_count += 1

    # ⏱️ 3. CALCULATE AVERAGES
    process_ms = round((time.time() - start_process) * 1000, 2)
    avg_per_post = round(process_ms / max(1, processed_count), 2)

    print("-" * 50)
    print(f"✅ COLLECTOR RUN COMPLETE")
    print(f"⏱️ Total Fetch Time:      {fetch_ms} ms")
    print(f"⏱️ Total Processing Time: {process_ms} ms")
    print(f"⚡ Average API Latency:   {avg_per_post} ms / post")
    print("=" * 50 + "\n")

    return {
        "total_fetched": len(data),
        "processed": processed_count,
        "latency_metrics": {
            "fetch_ms": fetch_ms,
            "processing_ms": process_ms,
            "avg_per_post_ms": avg_per_post
        },
        "results": results[:5]   # limit preview for API response
    }