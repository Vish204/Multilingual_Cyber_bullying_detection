import praw
import requests
import os
from dotenv import load_dotenv
import datetime

from cyberbullying.collector.cleaning_utils import clean_text
from cyberbullying.collector.language_utils import compute_language_distribution
from cyberbullying.collector.balancing import needs_balancing, get_missing_languages

load_dotenv()

API_URL = os.getenv("API_URL")

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)


# ---------------------------
# 🔹 Natural Posts
# ---------------------------
def fetch_reddit_posts(limit=2):

    data = []
    subreddit = reddit.subreddit("india")

    for post in subreddit.hot(limit=limit):

        text = (post.title or "") + " " + (post.selftext or "")
        text = clean_text(text)

        if text:
            data.append({
                "platform_post_id": f"t3_{post.id}", # Reddit uses t3_ prefix for posts
                "platform_time": datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc).isoformat(),
                "text": text,
                "platform": "reddit",
                "content_type": "post"
            })

    return data


# ---------------------------
# 🔹 Natural Comments
# ---------------------------
def fetch_reddit_comments(post_limit=2, comment_limit=2):

    data = []
    subreddit = reddit.subreddit("india")

    for submission in subreddit.hot(limit=post_limit):

        submission.comments.replace_more(limit=0)

        for comment in submission.comments.list()[:comment_limit]:

            text = clean_text(comment.body)

            if text:
                data.append({
                    "platform_post_id": f"t1_{comment.id}", # t1_ for comments
                    "platform_time": datetime.datetime.fromtimestamp(comment.created_utc, tz=datetime.timezone.utc).isoformat(),
                    "text": text,
                    "platform": "reddit",
                    "content_type": "comment"
                })

    return data


# ---------------------------
# 🔹 Targeted (Multilingual)
# ---------------------------
LANGUAGE_QUERIES = {
    "hindi": "हिंदी",
    "marathi": "मराठी",
    "tamil": "தமிழ்",
    "bengali": "বাংলা",
    "gujarati": "ગુજરાતી",
    "kannada": "ಕನ್ನಡ",
    "telugu": "తెలుగు",
    "malayalam": "മലയാളം",
    "punjabi": "ਪੰਜਾਬੀ",
    "urdu": "اردو"
}


def fetch_targeted_reddit(language, limit=3):

    query = LANGUAGE_QUERIES.get(language)
    if not query:
        return []

    data = []
    subreddit = reddit.subreddit("all")

    for post in subreddit.search(query, limit=limit):

        text = clean_text((post.title or "") + " " + (post.selftext or ""))

        if text:
            data.append({
                "platform_post_id": f"t3_{post.id}", # 🔥 FIXED
                "platform_time": datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc).isoformat(), # 🔥 FIXED
                "text": text,
                "platform": "reddit",
                "content_type": "post"
            })

    return data



# ---------------------------
# 🔹 Send to API
# ---------------------------
def send_to_api(item):

    payload = {
        "text": item["text"],
        "platform": item["platform"],
        "content_type": item["content_type"],
        "platform_post_id": item.get("platform_post_id"), 
        "platform_time": item.get("platform_time")
    }

    try:
        response = requests.post(API_URL, json=payload )
        return response.json()

    except requests.exceptions.Timeout:
        return {"error": "timeout"}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# 🔹 FINAL PIPELINE
# ---------------------------
def fetch_all_reddit_content():

    data = []

    data.extend(fetch_reddit_posts())
    data.extend(fetch_reddit_comments())

    # Language distribution
    lang_count = compute_language_distribution(data)

    # Balance
    if needs_balancing(lang_count):
        missing = get_missing_languages(lang_count)

        for lang in missing:
            data.extend(fetch_targeted_reddit(lang))

    return data