from googleapiclient.discovery import build
import pandas as pd, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def collect(language, keywords_batch, api_keys):
    print(f"--- YouTube Collector: Processing {len(keywords_batch)} keywords for {language.title()} ---")
    youtube = build('youtube', 'v3', developerKey=api_keys['youtube']['api_key'])
    collected_data = []

    for keyword in keywords_batch:
        print(f"   Searching for: '{keyword}'...")
        try:
            video_search = youtube.search().list(q=keyword, part="snippet", type="video", maxResults=2).execute()
            video_ids = [item['id']['videoId'] for item in video_search.get('items', [])]
            for video_id in video_ids:
                comment_threads = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", maxResults=5).execute()
                for item in comment_threads.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    collected_data.append({'platform': 'YouTube', 'text': comment['textDisplay']})
        except Exception as e:
            print(f"   ⚠️ Could not search for '{keyword}': {e}. (Quota issue?).")

    if collected_data:
        df = pd.DataFrame(collected_data)
        output_dir = os.path.join('data', 'raw', 'youtube', language)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'data.csv')
        df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path), encoding='utf-8-sig')
        print(f"✅ Appended {len(df)} new comments.")