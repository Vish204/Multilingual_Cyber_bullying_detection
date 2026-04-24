import requests
import os
from dotenv import load_dotenv
import random

from cyberbullying.collector.cleaning_utils import clean_text
from cyberbullying.collector.language_utils import compute_language_distribution
from cyberbullying.collector.balancing import needs_balancing, get_missing_languages

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
API_URL = os.getenv("API_URL")

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


# ---------------------------
# 🔹 Fetch Videos
# ---------------------------
def fetch_videos(query="bollywood controversy OR bigg boss fight OR worst umpiring OR overrated", max_results=5):

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": API_KEY
    }

    response = requests.get(SEARCH_URL, params=params )
    data = response.json()

    return [item["id"]["videoId"] for item in data.get("items", [])]


# ---------------------------
# 🔹 Fetch Comments
# ---------------------------
def fetch_comments(video_id, limit=3):

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": limit,
        "key": API_KEY,
        "textFormat": "plainText"
    }

    response = requests.get(COMMENTS_URL, params=params )
    data = response.json()

    comments = []

    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        text = clean_text(snippet["textDisplay"])

        if text:
            comments.append({
                "platform_post_id": item["id"], 
                "platform_time": snippet["publishedAt"], # YouTube already gives ISO format!
                "text": text,
                "platform": "youtube",
                "content_type": "comment"
            })

    return comments


# ---------------------------
# 🔹 Natural
# ---------------------------
def fetch_youtube_comments():

    data = []

    video_ids = fetch_videos()

    for vid in video_ids:
        data.extend(fetch_comments(vid))

    return data


# ---------------------------
# 🔹 Targeted
# ---------------------------
LANGUAGE_QUERIES = {
    "hindi": "  bakwas OR chutiya OR pagal",
    "marathi": "  फालतू OR मूर्ख",
    "tamil": "  முட்டாள் OR மோசமான",
    "bengali": "  বাজে OR বোকা",
    "gujarati": "  બકવાસ OR મૂર્ખ",
    "kannada": "  ಕೆಟ್ಟ OR ದಡ್ಡ",
    "telugu": "  చెత్త OR మూర్ఖుడు",
    "malayalam": "  മോശം OR വിഡ്ഢി",
    "punjabi": "  ਬਕਵਾਸ OR ਮੂਰਖ",
    "urdu": "  بکواس OR پاگل"
}


def fetch_targeted_youtube(language, limit=3):

    query = LANGUAGE_QUERIES.get(language)
    if not query:
        return []

    data = []
#Earlier 
    # video_ids = fetch_videos(query)

    # for vid in video_ids:
    #     data.extend(fetch_comments(vid))

    # return data

#  PAGINATION FIX: Fetch 15 videos, shuffle them, and dig for comments
    video_ids = fetch_videos(query, max_results=15)
    random.shuffle(video_ids)

    for vid in video_ids:
        # Get comments from this random video
        comments = fetch_comments(vid, limit=5)
        data.extend(comments)
        
        # Stop digging once we hit our limit of 3!
        if len(data) >= limit:
            break

    return data[:limit]





# ---------------------------
# 🔹 Send
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
def fetch_all_youtube_content():

    data = fetch_youtube_comments()

    lang_count = compute_language_distribution(data)

    if needs_balancing(lang_count):
        missing = get_missing_languages(lang_count)

        for lang in missing:
            data.extend(fetch_targeted_youtube(lang))

    return data