# run_preprocessing_final_fixed.py
import sys
import os
from pathlib import Path
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_keywords_corrected():
    """Load keywords correctly from complete database"""
    db_file = Path("resources/keywords/complete_multilingual_database.json")
    
    if not db_file.exists():
        logger.error("Complete database not found!")
        return {}
    
    try:
        with open(db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_keywords = {}
        
        if 'languages' in data:
            for lang, lang_data in data['languages'].items():
                if 'keywords' in lang_data and isinstance(lang_data['keywords'], list):
                    keywords = lang_data['keywords']
                    # Clean and filter
                    cleaned_keywords = []
                    for kw in keywords:
                        if isinstance(kw, str) and kw.strip():
                            cleaned_keywords.append(kw.strip().lower())
                    
                    all_keywords[lang] = cleaned_keywords
                    logger.info(f"Loaded {len(cleaned_keywords)} keywords for {lang}")
        
        return all_keywords
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {}

def create_labeling_patterns(keywords_dict, max_keywords=200):
    """Create regex patterns"""
    patterns = {}
    
    for lang, keywords in keywords_dict.items():
        if keywords:
            try:
                # Use first N keywords
                keywords_to_use = keywords[:max_keywords]
                escaped_keywords = [re.escape(kw) for kw in keywords_to_use]
                pattern = r'(?i)\b(' + '|'.join(escaped_keywords) + r')\b'
                patterns[lang] = re.compile(pattern)
                logger.debug(f"Pattern for {lang}: {len(keywords_to_use)} keywords")
            except Exception as e:
                logger.error(f"Pattern error for {lang}: {e}")
    
    return patterns

def load_raw_data():
    """
    Load all raw data from Twitter, Reddit, YouTube, and hard_toxic.
    Supports language subfolders.
    Preserves existing labels.
    """

    raw_dir = Path("data/raw")
    all_data = []

    platforms = ['twitter', 'reddit', 'youtube', 'hard_toxic']

    for platform in platforms:
        platform_dir = raw_dir / platform

        if not platform_dir.exists():
            continue

        logger.info(f"Loading from {platform}...")

        # ===============================
        # 🔴 HARD TOXIC (no language folders)
        # ===============================
        if platform == "hard_toxic":

            for csv_file in platform_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)

                    # Detect text column flexibly
                    possible_cols = ['text', 'tweet', 'comment', 'content', 'post']
                    text_col = None

                    for col in df.columns:
                        if col.lower() in possible_cols:
                            text_col = col
                            break

                    if not text_col:
                        logger.warning(f"No text column in {csv_file.name}")
                        continue

                    df = df.rename(columns={text_col: 'text'})

                    # Preserve label if exists
                    if 'label' in df.columns:
                        df = df[['text', 'label']]
                    else:
                        df = df[['text']]

                    df['language'] = df.get('language', 'english')
                    df['platform'] = 'synthetic'
                    df['source'] = str(csv_file)

                    all_data.append(df)
                    logger.info(f"  hard_toxic: {len(df)} rows from {csv_file.name}")

                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")

        # ===============================
        # 🔵 NORMAL DATASETS (with language folders)
        # ===============================
        else:
            for language_folder in platform_dir.iterdir():

                if not language_folder.is_dir():
                    continue

                language = language_folder.name

                for csv_file in language_folder.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)

                        # Flexible text column detection
                        text_col = None
                        for col in df.columns:
                            col_lower = col.lower()
                            if any(word in col_lower for word in 
                                   ['text', 'content', 'comment', 'tweet', 'post']):
                                text_col = col
                                break

                        if not text_col:
                            logger.warning(f"No text column in {csv_file.name}")
                            continue

                        df = df.rename(columns={text_col: 'text'})

                        # Preserve label if exists
                        if 'label' in df.columns:
                            df = df[['text', 'label']]
                        else:
                            df = df[['text']]

                        df['language'] = language
                        df['platform'] = platform
                        df['source'] = str(csv_file)

                        all_data.append(df)
                        logger.info(f"  {language}: {len(df)} rows from {csv_file.name}")

                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")

    if not all_data:
        logger.error("No data loaded!")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total loaded: {len(combined_df)} rows")

    return combined_df


def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Basic cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Mentions/hashtags
    text = re.sub(r'[^\w\s@#]', ' ', text)  # Special chars
    text = ' '.join(text.split())  # Extra whitespace
    
    return text.lower().strip()

def main():
    """Main preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("PREPROCESSING WITH CORRECT KEYWORDS")
    logger.info("=" * 60)
    
    # Step 1: Load keywords
    logger.info("1. Loading keywords...")
    all_keywords = load_keywords_corrected()
    
    if not all_keywords:
        logger.error("Failed to load keywords!")
        return False
    
    logger.info(f"✓ Loaded {len(all_keywords)} languages")
    
    # Step 2: Create labeling patterns
    logger.info("2. Creating labeling patterns...")
    patterns = create_labeling_patterns(all_keywords, max_keywords=200)
    
    # Step 3: Load raw data
    logger.info("3. Loading raw data...")
    raw_df = load_raw_data()
    
    if raw_df.empty:
        logger.error("No raw data found!")
        return False
    
    logger.info(f"✓ Loaded {len(raw_df)} raw samples")
    logger.info(f"  Languages: {raw_df['language'].nunique()}")
    logger.info(f"  Platforms: {raw_df['platform'].nunique()}")
    
    # Step 4: Clean text
    logger.info("4. Cleaning text...")
    raw_df['cleaned_text'] = raw_df['text'].apply(clean_text)
    
    # Remove empty texts
    initial_count = len(raw_df)
    raw_df = raw_df[raw_df['cleaned_text'].str.strip() != '']
    logger.info(f"  After cleaning: {len(raw_df)} rows (removed {initial_count - len(raw_df)} empty)")
    
    # Remove duplicate texts BEFORE labeling and splitting (CRITICAL - prevents leakage)
    logger.info("Removing duplicate texts...")

    before = len(raw_df)

    raw_df = raw_df.drop_duplicates(subset=["cleaned_text"])

    after = len(raw_df)

    logger.info(f"Removed {before - after} duplicate samples")
    logger.info(f"Remaining samples: {after}")


    # Step 5: Label data using multilingual keywords
    logger.info("5. Labeling data...")
    
    def label_text(row):
        text = str(row['cleaned_text'])
        language = row['language']
        
        # Try language-specific pattern
        if language in patterns:
            if patterns[language].search(text):
                return 1
        
        # Fallback to English pattern
        if 'english' in patterns:
            if patterns['english'].search(text):
                return 1
        
        return 0
    
    
    # Only label rows where label is missing
    if 'label' not in raw_df.columns:
        raw_df['label'] = raw_df.apply(label_text, axis=1)
    else:
        raw_df['label'] = raw_df.apply(
            lambda row: row['label'] if pd.notna(row['label']) else label_text(row),
            axis=1
        )

    
    # Show statistics
    label_counts = raw_df['label'].value_counts()
    logger.info(f"  Label distribution: {dict(label_counts)}")
    logger.info(f"  Bullying percentage: {label_counts.get(1, 0)/len(raw_df)*100:.2f}%")
    
    # Language distribution
    logger.info("\n  Language-wise distribution:")
    for lang in sorted(raw_df['language'].unique()):
        lang_df = raw_df[raw_df['language'] == lang]
        bullying_count = len(lang_df[lang_df['label'] == 1])
        total_count = len(lang_df)
        percentage = bullying_count/total_count*100 if total_count > 0 else 0
        logger.info(f"    {lang:12}: {total_count:4} samples, {bullying_count:3} bullying ({percentage:5.1f}%)")
    
    # Step 6: Split and save data
    logger.info("6. Splitting and saving data...")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # For each language, do stratified split
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for language in raw_df['language'].unique():
        lang_df = raw_df[raw_df['language'] == language]
        
        if len(lang_df) < 10:
            # Too few samples, put all in train
            train_dfs.append(lang_df)
            continue
        
        try:
            # Try stratified split
            lang_train, lang_temp = train_test_split(
                lang_df, 
                test_size=0.3, 
                random_state=42,
                stratify=lang_df['label']
            )
            
            lang_val, lang_test = train_test_split(
                lang_temp, 
                test_size=0.5, 
                random_state=42,
                stratify=lang_temp['label']
            )
            
            train_dfs.append(lang_train)
            val_dfs.append(lang_val)
            test_dfs.append(lang_test)
            
        except Exception as e:
            logger.warning(f"Stratified split failed for {language}: {e}")
            # If stratification fails, do random split
            lang_train, lang_temp = train_test_split(lang_df, test_size=0.3, random_state=42)
            lang_val, lang_test = train_test_split(lang_temp, test_size=0.5, random_state=42)
            
            train_dfs.append(lang_train)
            val_dfs.append(lang_val)
            test_dfs.append(lang_test)
    
    # Combine all
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Drop original raw text column
    train_df = train_df.drop(columns=['text'])
    val_df = val_df.drop(columns=['text'])
    test_df = test_df.drop(columns=['text'])

    # Rename cleaned_text to text
    train_df = train_df.rename(columns={'cleaned_text': 'text'})
    val_df = val_df.rename(columns={'cleaned_text': 'text'})
    test_df = test_df.rename(columns={'cleaned_text': 'text'})

    
    # Keep only necessary columns
    keep_cols = ['text', 'label', 'language', 'platform']
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]
    test_df = test_df[keep_cols]
    
    # -------------------------------------------------
    # Leakage safety check (VERY IMPORTANT)
    # -------------------------------------------------
    logger.info("Running leakage safety check...")

    train_set = set(train_df['text'])
    val_set = set(val_df['text'])
    test_set = set(test_df['text'])

    logger.info(f"Train-Val overlap: {len(train_set & val_set)}")
    logger.info(f"Train-Test overlap: {len(train_set & test_set)}")
    logger.info(f"Val-Test overlap: {len(val_set & test_set)}")




    # Save
    train_df.to_csv(processed_dir / "train_data.csv", index=False)
    val_df.to_csv(processed_dir / "val_data.csv", index=False)
    test_df.to_csv(processed_dir / "test_data.csv", index=False)
    
    # Save combined dataset
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(processed_dir / "multilingual_dataset.csv", index=False)
    
    # Save statistics
    stats = {
        "total_samples": len(combined_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "languages": combined_df['language'].nunique(),
        "platforms": combined_df['platform'].nunique(),
        "label_distribution": combined_df['label'].value_counts().to_dict(),
        "language_distribution": combined_df['language'].value_counts().to_dict(),
        "platform_distribution": combined_df['platform'].value_counts().to_dict()
    }
    
    with open(processed_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"✓ Total samples: {len(combined_df)}")
    logger.info(f"✓ Train: {len(train_df)}")
    logger.info(f"✓ Validation: {len(val_df)}")
    logger.info(f"✓ Test: {len(test_df)}")
    logger.info(f"✓ Languages: {combined_df['language'].nunique()}")
    logger.info(f"✓ Saved to: {processed_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)