import requests
import os
from dotenv import load_dotenv

from cyberbullying.collector.cleaning_utils import clean_text
from cyberbullying.collector.language_utils import compute_language_distribution
from cyberbullying.collector.balancing import needs_balancing, get_missing_languages

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
API_URL = os.getenv("API_URL")

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}


# ---------------------------
# 🔹 Natural Tweets
# ---------------------------
def fetch_tweets(limit=6):

    params = {
        "query": "india OR mumbai OR delhi -is:retweet",
        "max_results": min(limit, 100)
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params)
    data = response.json()

    tweets = []

    for tweet in data.get("data", []):
        text = clean_text(tweet.get("text", ""))

        if text:
            tweets.append({
                "text": text,
                "platform": "twitter",
                "content_type": "tweet"
            })

    return tweets


# ---------------------------
# 🔹 Targeted
# ---------------------------
LANGUAGE_KEYWORDS = {
    "marathi": "मराठी OR महाराष्ट्र",
    "hindi": "हिंदी OR भारत",
    "tamil": "தமிழ் OR சென்னை",
    "bengali": "বাংলা OR কলকাতা"
}


def fetch_targeted_tweets(language, limit=3):

    query = LANGUAGE_KEYWORDS.get(language)
    if not query:
        return []

    params = {
        "query": f"{query} -is:retweet",
        "max_results": limit
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params)
    data = response.json()

    tweets = []

    for tweet in data.get("data", []):
        text = clean_text(tweet.get("text", ""))

        if text:
            tweets.append({
                "text": text,
                "platform": "twitter",
                "content_type": "tweet"
            })

    return tweets


# ---------------------------
# 🔹 Dedup
# ---------------------------
seen_hashes = set()

def is_duplicate(text):
    key = text.strip().lower()

    if key in seen_hashes:
        return True

    seen_hashes.add(key)
    return False


# ---------------------------
# 🔹 Send to API
# ---------------------------
def send_to_api(item):

    payload = {
        "text": item["text"],
        "platform": item["platform"],
        "content_type": item["content_type"]
    }

    try:
        response = requests.post(API_URL, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# 🔹 FINAL PIPELINE
# ---------------------------
def fetch_all_twitter_content():

    data = fetch_tweets()

    lang_count = compute_language_distribution(data)

    if needs_balancing(lang_count):
        missing = get_missing_languages(lang_count)

        for lang in missing:
            data.extend(fetch_targeted_tweets(lang))

    return data