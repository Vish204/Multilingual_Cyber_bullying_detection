def needs_balancing(lang_count, threshold=0.6):

    total = sum(lang_count.values())

    if total == 0:
        return False

    english_ratio = lang_count.get("english", 0) / total

    return english_ratio > threshold


def get_missing_languages(lang_count):

    target_languages = ["marathi", "hindi", "tamil", "bengali"]

    missing = []

    for lang in target_languages:
        if lang not in lang_count:
            missing.append(lang)

    return missing