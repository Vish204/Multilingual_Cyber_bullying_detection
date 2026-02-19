# load_keywords_corrected.py
import json
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_keywords_from_complete_db():
    """Load ALL keywords from complete database (1300+ per language)"""
    db_file = Path("config/complete_multilingual_database.json")
    
    if not db_file.exists():
        logger.error(f"Complete database not found: {db_file}")
        return {}
    
    try:
        with open(db_file, 'r') as f:
            data = json.load(f)
        
        all_keywords = {}
        
        # Extract from languages section
        if 'languages' in data:
            for lang, lang_data in data['languages'].items():
                if 'keywords' in lang_data and isinstance(lang_data['keywords'], list):
                    keywords = lang_data['keywords']
                    # Clean keywords
                    cleaned_keywords = []
                    for kw in keywords:
                        if isinstance(kw, str) and kw.strip():
                            cleaned_keywords.append(kw.strip().lower())
                    
                    all_keywords[lang] = cleaned_keywords
                    logger.info(f"Loaded {len(cleaned_keywords)} keywords for {lang}")
        
        return all_keywords
        
    except Exception as e:
        logger.error(f"Error loading complete database: {e}")
        return {}

def load_keywords_from_individual_files():
    """Load keywords from individual files (200 per language)"""
    keywords_dir = Path("config/multilingual_keywords")
    all_keywords = {}
    
    for keyword_file in keywords_dir.glob("keywords_*.json"):
        lang = keyword_file.stem.replace("keywords_", "")
        
        try:
            with open(keyword_file, 'r') as f:
                data = json.load(f)
            
            # Extract keywords from the 'keywords' key
            if isinstance(data, dict) and 'keywords' in data:
                keywords = data['keywords']
                if isinstance(keywords, list):
                    # Clean keywords
                    cleaned_keywords = []
                    for kw in keywords:
                        if isinstance(kw, str) and kw.strip():
                            cleaned_keywords.append(kw.strip().lower())
                    
                    all_keywords[lang] = cleaned_keywords
                    logger.info(f"Loaded {len(cleaned_keywords)} keywords for {lang} from individual file")
            
        except Exception as e:
            logger.error(f"Error loading {lang}: {e}")
    
    return all_keywords

def create_labeling_patterns(keywords_dict, max_keywords=200):
    """Create regex patterns for each language"""
    patterns = {}
    
    for lang, keywords in keywords_dict.items():
        if keywords:
            try:
                # Use first N keywords to avoid huge regex
                keywords_to_use = keywords[:max_keywords]
                
                # Escape special regex characters
                escaped_keywords = [re.escape(kw) for kw in keywords_to_use]
                
                # Create pattern
                pattern = r'(?i)\b(' + '|'.join(escaped_keywords) + r')\b'
                patterns[lang] = re.compile(pattern)
                
                logger.info(f"Created pattern for {lang} with {len(keywords_to_use)} keywords")
                
            except Exception as e:
                logger.error(f"Error creating pattern for {lang}: {e}")
    
    return patterns

def main():
    logger.info("=" * 60)
    logger.info("LOADING KEYWORDS - CORRECTED VERSION")
    logger.info("=" * 60)
    
    # Try to load from complete database first
    logger.info("\n1. Loading from complete database...")
    complete_keywords = load_keywords_from_complete_db()
    
    if complete_keywords:
        logger.info(f"✓ Loaded {len(complete_keywords)} languages from complete DB")
        for lang, keywords in complete_keywords.items():
            logger.info(f"   {lang}: {len(keywords)} keywords")
    
    # Also load from individual files
    logger.info("\n2. Loading from individual files...")
    individual_keywords = load_keywords_from_individual_files()
    
    if individual_keywords:
        logger.info(f"✓ Loaded {len(individual_keywords)} languages from individual files")
        for lang, keywords in individual_keywords.items():
            logger.info(f"   {lang}: {len(keywords)} keywords")
    
    # Merge (prefer complete DB if available)
    all_keywords = {}
    if complete_keywords:
        all_keywords = complete_keywords
    elif individual_keywords:
        all_keywords = individual_keywords
    
    # Create patterns
    logger.info("\n3. Creating labeling patterns...")
    patterns = create_labeling_patterns(all_keywords, max_keywords=200)
    
    logger.info(f"\n✓ Total languages: {len(patterns)}")
    
    # Test patterns
    logger.info("\n4. Testing patterns:")
    test_texts = {
        'english': "You are stupid and idiot",
        'hindi': "तू मूर्ख है",
        'marathi': "तू मूर्ख आहेस"
    }
    
    for lang, text in test_texts.items():
        if lang in patterns:
            matches = patterns[lang].findall(text.lower())
            logger.info(f"   {lang}: '{text}' → Matches: {matches}")
        else:
            logger.warning(f"   {lang}: No pattern available")
    
    return all_keywords, patterns

if __name__ == "__main__":
    keywords, patterns = main()