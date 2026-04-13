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
        # "query": "(india OR mumbai OR delhi OR भारत OR मुंबई OR दिल्ली OR தமிழ் OR বাংলা OR मराठी) -is:retweet",
        "query": "(boycott OR scam OR overrated OR worst OR fake OR nepotism OR bakwas OR chhapri) -is:retweet",
        "max_results": min(limit, 50),
        "tweet.fields": "lang,text,created_at,id"
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params )

    print("Twitter status:", response.status_code)
    print("Twitter response:", response.text[:500])
    data = response.json()

    tweets = []

    print("Tweets fetched:", len(data.get("data", [])))

    for tweet in data.get("data", []):
        text = clean_text(tweet.get("text", ""))


        if text:
            tweets.append({
                "platform_post_id": tweet.get("id"),                 
                "platform_time": tweet.get("created_at"),
                "text": text,
                "platform": "twitter",  
                "content_type": "tweet"
            })

    #  FALLBACK (if empty)
    if not tweets:
        print("Twitter fallback triggered")

        fallback_params = {
            "query": "(stupid OR hate OR complain OR blocked OR angry) -is:retweet",
            "max_results": 20,
            "tweet.fields": "lang,text"
        }

        try:
            response = requests.get(
                SEARCH_URL,
                headers=headers,
                params=fallback_params,
                timeout=5
            )

            data = response.json()

            for tweet in data.get("data", []):
                text = clean_text(tweet.get("text", ""))

                if text:
                    tweets.append({
                        "text": text,
                        "platform": "twitter",
                        "content_type": "tweet"
                    })

        except Exception as e:
            print("Twitter fallback error:", e)
    return tweets


# ---------------------------
# 🔹 Targeted
# ---------------------------
LANGUAGE_QUERIES = {
    "hindi": "idiot OR stupid OR bakwas OR chutiya OR pagal",
    "marathi": "idiot OR stupid OR फालतू OR मूर्ख",
    "tamil": "idiot OR stupid OR முட்டாள் OR மோசமான",
    "bengali": "idiot OR stupid OR বাজে OR বোকা",
    "gujarati": "idiot OR stupid OR બકવાસ OR મૂર્ખ",
    "kannada": "idiot OR stupid OR ಕೆಟ್ಟ OR ದಡ್ಡ",
    "telugu": "idiot OR stupid OR చెత్త OR మూర్ఖుడు",
    "malayalam": "idiot OR stupid OR മോശം OR വിഡ്ഢി",
    "punjabi": "idiot OR stupid OR ਬਕਵਾਸ OR ਮੂਰਖ",
    "urdu": "idiot OR stupid OR بکواس OR پاگل"
}


def fetch_targeted_tweets(language, limit=3):

    query = LANGUAGE_KEYWORDS.get(language)
    if not query:
        return []

    params = {
        "query": f"{query} -is:retweet",
        "max_results": limit,
        "tweet.fields": "lang,text,created_at,id"
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params )
    data = response.json()

    tweets = []

    for tweet in data.get("data", []):
        text = clean_text(tweet.get("text", ""))

        if text:
            tweets.append({
                "platform_post_id": tweet.get("id"),         
                "platform_time": tweet.get("created_at"),
                "text": text,
                "platform": "twitter",
                "content_type": "tweet"
            })

    return tweets



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
def fetch_all_twitter_content():

    data = fetch_tweets()

    lang_count = compute_language_distribution(data)

    if needs_balancing(lang_count):
        missing = get_missing_languages(lang_count)

        for lang in missing:
            data.extend(fetch_targeted_tweets(lang))

    return data