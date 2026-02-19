import tweepy, pandas as pd, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def collect(language, keywords_batch, api_keys):
    print(f"--- Twitter Collector: Processing {len(keywords_batch)} keywords for {language.title()} ---")
    client = tweepy.Client(bearer_token=api_keys['twitter']['bearer_token'], wait_on_rate_limit=True)
    collected_data = []

    query = " OR ".join([f'"{k}"' for k in keywords_batch]) + " -is:retweet"
    print(f"   Searching for batch: [{', '.join(keywords_batch[:2])}...]")

    try:
        response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at", "lang"])
        if response.data:
            for tweet in response.data:
                collected_data.append({'platform': 'X', 'language': language, 'text': tweet.text})
    except Exception as e:
        print(f"   ⚠️ Could not search batch: {e}")

    if collected_data:
        df = pd.DataFrame(collected_data)
        output_dir = os.path.join('data', 'raw', 'twitter', language)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'data.csv')
        df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path), encoding='utf-8-sig')
        print(f"✅ Appended {len(df)} new tweets.")