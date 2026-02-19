import praw, pandas as pd, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def collect(language, keywords_batch, api_keys):
    print(f"--- Reddit Collector: Processing {len(keywords_batch)} keywords for {language.title()} ---")
    reddit = praw.Reddit(client_id=api_keys['reddit']['client_id'], client_secret=api_keys['reddit']['client_secret'], user_agent=api_keys['reddit']['user_agent'])
    subreddit = reddit.subreddit("india+indiasocial+mumbai")
    collected_data = []

    for keyword in keywords_batch:
        print(f"   Searching for: '{keyword}'...")
        try:
            for submission in subreddit.search(keyword, limit=20):
                if keyword.lower() in submission.title.lower():
                    collected_data.append({'platform': 'Reddit', 'text': submission.title})
                for top_level_comment in submission.comments:
                    if isinstance(top_level_comment, praw.models.Comment) and keyword.lower() in top_level_comment.body.lower():
                        collected_data.append({'platform': 'Reddit', 'text': top_level_comment.body})
        except Exception as e:
            print(f"   ⚠️ Could not search for '{keyword}': {e}")

    if collected_data:
        df = pd.DataFrame(collected_data)
        output_dir = os.path.join('data', 'raw', 'reddit', language)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'data.csv')
        df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path), encoding='utf-8-sig')
        print(f"✅ Appended {len(df)} new items.")