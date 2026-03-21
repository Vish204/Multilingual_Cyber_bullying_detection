from cyberbullying.inference.inference_service import detect_language


def get_language_name(text):

    lang = detect_language(text)

    if isinstance(lang, dict):
        return lang.get("name", "unknown")

    return "unknown"


def compute_language_distribution(data):

    lang_count = {}

    for item in data:
        lang = get_language_name(item["text"])
        lang_count[lang] = lang_count.get(lang, 0) + 1

    return lang_count