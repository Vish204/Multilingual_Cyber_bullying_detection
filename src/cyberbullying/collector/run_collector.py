import time
from cyberbullying.collector.reddit_collector import fetch_all_content, is_duplicate, send_to_api


def run_pipeline():

    print("🚀 Starting Reddit Monitoring...\n")

    while True:

        try:
            data = fetch_all_content()

            for item in data:

                if is_duplicate(item["text"]):
                    continue

                print("\nTEXT:", item["text"][:100])

                result = send_to_api(item)

                print("RESULT:", result)

            print("\n⏳ Sleeping for 60 seconds...\n")
            time.sleep(60)

        except Exception as e:
            print("ERROR:", e)
            time.sleep(10)


if __name__ == "__main__":
    run_pipeline()