#!/usr/bin/env python3
"""
Build massive multilingual keyword database from 15,000+ English words
WITH REAL GOOGLE TRANSLATE - INCLUDES ENGLISH & HINGLISH
"""

import sys
import os
import json
import time
import concurrent.futures
from datetime import datetime
from deep_translator import GoogleTranslator

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_simple_logger():
    """Simple logger for the build process"""
    class SimpleLogger:
        def __init__(self):
            self.start_time = datetime.now()
            
        def info(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ℹ️  {message}")
            
        def error(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ❌ {message}")
            
        def success(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ {message}")
            
        def warning(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ⚠️  {message}")
    
    return SimpleLogger()

class RealTranslator:
    """
    REAL translator using Google Translate API - INCLUDES ENGLISH & HINGLISH
    """
    
    def __init__(self):
        self.logger = setup_simple_logger()
        
        # All languages including English and Hinglish
        self.all_languages = {
            'english': 'en',  # English added
            'hinglish': 'hi', # Hinglish uses Hindi base
            'hindi': 'hi',
            'bengali': 'bn', 
            'tamil': 'ta',
            'telugu': 'te',
            'marathi': 'mr',
            'gujarati': 'gu',
            'kannada': 'kn',
            'malayalam': 'ml',
            'punjabi': 'pa',
            'oriya': 'or',
            'urdu': 'ur',
            'sanskrit': 'sa'
        }
        
        self.translation_cache = {}
        self.failed_translations = []
        self.checkpoint_file = 'config/translation_checkpoint.json'
        self.resume_from_checkpoint = False
        self.checkpoint_data = {}
        
        # Speed optimization settings
        self.max_workers = 3
        self.batch_size = 5
        self.delay_between_requests = 0.05
    
    def load_checkpoint(self):
        """Load checkpoint data if exists"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                self.logger.success(f"📌 Checkpoint found! Resuming from word #{self.checkpoint_data.get('last_processed_word', 0)}")
                self.resume_from_checkpoint = True
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            return False
    
    def save_checkpoint(self, main_category, subcategory, word_index, total_processed, multilingual_db):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                'main_category': main_category,
                'subcategory': subcategory,
                'word_index': word_index,
                'last_processed_word': total_processed,
                'timestamp': datetime.now().isoformat(),
                'translation_cache': self.translation_cache,
                'failed_translations': self.failed_translations
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint: {e}")
            return False
    
    def cleanup_checkpoint(self):
        """Remove checkpoint files after successful completion"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            self.logger.info("🧹 Checkpoint files cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not clean up checkpoint files: {e}")
    
    def load_partial_database(self):
        """Load partial database from checkpoint"""
        partial_db_file = 'config/partial_multilingual_database.json'
        if os.path.exists(partial_db_file):
            try:
                with open(partial_db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load partial database: {e}")
        return None
    
    def save_partial_database(self, multilingual_db):
        """Save partial database for resume"""
        try:
            partial_db_file = 'config/partial_multilingual_database.json'
            with open(partial_db_file, 'w', encoding='utf-8') as f:
                json.dump(multilingual_db, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.warning(f"Could not save partial database: {e}")
            return False
    
    def initialize_database(self):
        """Initialize multilingual database structure"""
        return {
            'metadata': {
                'build_date': datetime.now().isoformat(),
                'total_english_words': 0,
                'languages_translated': list(self.all_languages.keys()),
                'translation_engine': 'Google Translate API + English/Hinglish',
                'note': 'Real translations from Google Translate + English base + Hinglish mix',
                'resumed_from_checkpoint': self.resume_from_checkpoint
            },
            'languages': {},
            'categories': {}
        }
    
    def load_english_database(self):
        """Load the massive English keyword database"""
        try:
            with open('config/english_keyword_base.json', 'r', encoding='utf-8') as f:
                english_db = json.load(f)
                
            # Count total words
            total_words = 0
            for main_category, subcategories in english_db.items():
                for subcategory, words in subcategories.items():
                    total_words += len(words)
                    
            self.logger.success(f"Loaded English database: {total_words} words across {len(english_db)} categories")
            return english_db
            
        except FileNotFoundError:
            self.logger.error("English keyword database not found at config/english_keyword_base.json")
            self.logger.info("Please create the file with your 15,000+ words")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in English database: {e}")
            return None
    
    def create_english_keywords_directly(self, english_db):
        """Create English keywords directly from English database (no translation needed)"""
        self.logger.info("🔤 Creating English keywords directly...")
        
        english_keywords = {
            'language': 'english',
            'total_keywords': 0,
            'keywords': [],
            'categories': {},
            'translation_success_rate': 100.0
        }
        
        all_keywords = set()
        categories_data = {}
        
        for main_category, subcategories in english_db.items():
            for subcategory, words in subcategories.items():
                if subcategory not in categories_data:
                    categories_data[subcategory] = []
                
                for word in words:
                    # For English, native and roman are the same as English word
                    entry = {
                        'english': word,
                        'native': word,
                        'roman': word,
                        'translation_success': True,
                        'category': main_category,
                        'subcategory': subcategory
                    }
                    categories_data[subcategory].append(entry)
                    all_keywords.add(word)
        
        english_keywords['keywords'] = list(all_keywords)
        english_keywords['total_keywords'] = len(english_keywords['keywords'])
        english_keywords['categories'] = categories_data
        
        self.logger.success(f"✅ English: {english_keywords['total_keywords']} keywords created")
        return english_keywords
    
    def create_hinglish_keywords(self, english_db, hindi_translations):
        """Create Hinglish keywords by mixing English and Hindi words"""
        self.logger.info("🔤 Creating Hinglish keywords (English + Hindi mix)...")
        
        hinglish_keywords = {
            'language': 'hinglish',
            'total_keywords': 0,
            'keywords': [],
            'categories': {},
            'translation_success_rate': 100.0
        }
        
        # Get English words
        english_words = set()
        for subcategory, words in english_db.items():
            for word_list in words.values():
                english_words.update(word_list[:50])  # Top 50 English words per category
        
        # Get Hindi words (from translations)
        hindi_words = set()
        hinglish_categories = {}
        
        if hindi_translations and 'categories' in hindi_translations:
            for subcategory, entries in hindi_translations['categories'].items():
                if subcategory not in hinglish_categories:
                    hinglish_categories[subcategory] = []
                
                # Take top 20 Hindi entries per subcategory
                for entry in entries[:20]:
                    hindi_words.add(entry['roman'])
                    hinglish_categories[subcategory].append(entry)
        
        # Mix English and Hindi words for Hinglish
        hinglish_words = set()
        hinglish_words.update(list(english_words)[:100])  # Top 100 English
        hinglish_words.update(list(hindi_words)[:100])    # Top 100 Hindi
        
        # Also add some mixed entries to categories
        for subcategory in list(hinglish_categories.keys())[:10]:  # Top 10 categories
            if subcategory in hinglish_categories:
                # Add some English entries to Hindi categories for true mixing
                english_entries = []
                for word in list(english_words)[:10]:
                    english_entries.append({
                        'english': word,
                        'native': word,
                        'roman': word,
                        'translation_success': True,
                        'category': 'mixed',
                        'subcategory': subcategory
                    })
                hinglish_categories[subcategory].extend(english_entries)
        
        hinglish_keywords['keywords'] = list(hinglish_words)
        hinglish_keywords['total_keywords'] = len(hinglish_keywords['keywords'])
        hinglish_keywords['categories'] = hinglish_categories
        
        self.logger.success(f"✅ Hinglish: {hinglish_keywords['total_keywords']} mixed keywords created")
        return hinglish_keywords
    
    def translate_batch_parallel(self, english_words, target_langs):
        """Translate a batch of words to multiple languages in parallel"""
        results = {}
        
        def translate_single_word(word, lang_name):
            """Translate single word - used for parallel processing"""
            cache_key = f"{word}_{lang_name}"
            
            # Check cache first
            if cache_key in self.translation_cache:
                return word, lang_name, self.translation_cache[cache_key]
            
            try:
                # Skip translation for English (words remain same)
                if lang_name == 'english':
                    result = {
                        'native': word,
                        'roman': word,
                        'success': True
                    }
                    self.translation_cache[cache_key] = result
                    return word, lang_name, result
                
                # Get language code
                lang_code = self.all_languages[lang_name]
                
                # Initialize translator
                translator = GoogleTranslator(source='en', target=lang_code)
                
                # Translate to native script
                native_translation = translator.translate(word)
                
                # Simple Romanization
                romanized = native_translation
                
                result = {
                    'native': native_translation,
                    'roman': romanized,
                    'success': True
                }
                
                # Cache the result
                self.translation_cache[cache_key] = result
                
                # Small delay to avoid rate limiting
                time.sleep(self.delay_between_requests)
                
                return word, lang_name, result
                
            except Exception as e:
                error_msg = f"Translation failed for '{word}' to {lang_name}: {e}"
                self.failed_translations.append(error_msg)
                
                # Fallback: return English word
                return word, lang_name, {
                    'native': word,
                    'roman': word,
                    'success': False,
                    'error': str(e)
                }
        
        # Process translations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare all translation tasks
            future_to_translation = {}
            for word in english_words:
                for lang_name in target_langs:
                    future = executor.submit(translate_single_word, word, lang_name)
                    future_to_translation[future] = (word, lang_name)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_translation):
                word, lang_name = future_to_translation[future]
                try:
                    result_word, result_lang, translation_result = future.result()
                    if lang_name not in results:
                        results[lang_name] = {}
                    results[lang_name][word] = translation_result
                except Exception as e:
                    self.logger.warning(f"Parallel translation failed for {word} to {lang_name}: {e}")
        
        return results
    
    def build_multilingual_database(self):
        """Build the complete multilingual database including English & Hinglish"""
        self.logger.info("📖 Loading English keyword database...")
        
        english_db = self.load_english_database()
        if not english_db:
            return None
        
        # Load checkpoint if exists
        self.load_checkpoint()
        
        # Initialize or load partial database
        if self.resume_from_checkpoint and self.checkpoint_data:
            multilingual_db = self.load_partial_database()
            if not multilingual_db:
                multilingual_db = self.initialize_database()
            
            # Restore cache and failed translations
            self.translation_cache = self.checkpoint_data.get('translation_cache', {})
            self.failed_translations = self.checkpoint_data.get('failed_translations', [])
            
            resume_category = self.checkpoint_data.get('main_category')
            resume_subcategory = self.checkpoint_data.get('subcategory')
            resume_word_index = self.checkpoint_data.get('word_index', 0)
            total_words_processed = self.checkpoint_data.get('last_processed_word', 0)
            
            self.logger.success(f"🔄 RESUMING FROM CHECKPOINT: {resume_category} -> {resume_subcategory} at word #{resume_word_index}")
        else:
            multilingual_db = self.initialize_database()
            resume_category = None
            resume_subcategory = None
            resume_word_index = 0
            total_words_processed = 0
        
        # Initialize language structures if not present
        for lang_name in self.all_languages.keys():
            if lang_name not in multilingual_db['languages']:
                multilingual_db['languages'][lang_name] = {
                    'total_keywords': 0,
                    'keywords': [],
                    'categories': {},
                    'translation_success_rate': 0
                }
        
        successful_translations = 0
        start_time = time.time()
        resume_found = not self.resume_from_checkpoint
        
        # Target languages (exclude English as we handle it separately)
        target_languages = [lang for lang in self.all_languages.keys() if lang not in ['english', 'hinglish']]
        
        self.logger.info(f"🚀 PROCESSING {len(target_languages)} LANGUAGES + ENGLISH + HINGLISH")
        self.logger.info(f"📊 Target languages: {', '.join(target_languages)}")
        
        # First, create English keywords directly (no translation needed)
        if 'english' not in multilingual_db['languages'] or not multilingual_db['languages']['english']['keywords']:
            english_keywords = self.create_english_keywords_directly(english_db)
            multilingual_db['languages']['english'] = english_keywords
        
        # Process translation for other languages
        for main_category, subcategories in english_db.items():
            if main_category not in multilingual_db['categories']:
                multilingual_db['categories'][main_category] = {}
            
            # Skip categories until we reach the resume point
            if not resume_found and self.resume_from_checkpoint:
                if main_category != resume_category:
                    self.logger.info(f"⏭️  Skipping category: {main_category} (already processed)")
                    continue
                resume_found = True
            
            self.logger.info(f"🔤 Translating category: {main_category}")
            
            for subcategory, english_words in subcategories.items():
                if subcategory not in multilingual_db['categories'][main_category]:
                    multilingual_db['categories'][main_category][subcategory] = {}
                
                # Skip subcategories until we reach the resume point
                if not resume_found and self.resume_from_checkpoint:
                    if subcategory != resume_subcategory:
                        self.logger.info(f"⏭️  Skipping subcategory: {subcategory} (already processed)")
                        continue
                    resume_found = True
                
                self.logger.info(f"   📋 Subcategory: {subcategory} ({len(english_words)} words)")
                
                # Process words in batches for better performance
                words_to_process = english_words[resume_word_index:] if not resume_found else english_words
                resume_word_index = 0
                
                for batch_start in range(0, len(words_to_process), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(words_to_process))
                    batch_words = words_to_process[batch_start:batch_end]
                    
                    if not batch_words:
                        continue
                    
                    # Translate batch in parallel
                    batch_results = self.translate_batch_parallel(batch_words, target_languages)
                    
                    # Process batch results
                    batch_success_count = 0
                    for english_word in batch_words:
                        word_success_count = 0
                        
                        for lang_name in target_languages:
                            if lang_name in batch_results and english_word in batch_results[lang_name]:
                                translation_result = batch_results[lang_name][english_word]
                                
                                # Add to language-specific database
                                if subcategory not in multilingual_db['languages'][lang_name]['categories']:
                                    multilingual_db['languages'][lang_name]['categories'][subcategory] = []
                                
                                translation_entry = {
                                    'english': english_word,
                                    'native': translation_result['native'],
                                    'roman': translation_result['roman'],
                                    'translation_success': translation_result['success'],
                                    'category': main_category,
                                    'subcategory': subcategory
                                }
                                
                                multilingual_db['languages'][lang_name]['categories'][subcategory].append(translation_entry)
                                multilingual_db['languages'][lang_name]['keywords'].append(translation_result['roman'])
                                
                                if translation_result['success']:
                                    word_success_count += 1
                                    batch_success_count += 1
                                    successful_translations += 1
                        
                        total_words_processed += 1
                    
                    # Save checkpoint and progress after each batch
                    self.save_checkpoint(main_category, subcategory, batch_end, total_words_processed, multilingual_db)
                    self.save_partial_database(multilingual_db)
                    
                    elapsed = time.time() - start_time
                    words_per_second = total_words_processed / elapsed if elapsed > 0 else 0
                    success_rate = (successful_translations / (total_words_processed * len(target_languages))) * 100
                    
                    self.logger.info(f"   📊 Progress: {total_words_processed} words "
                                   f"| Success: {success_rate:.1f}% "
                                   f"| Speed: {words_per_second:.2f} words/sec")
        
        # Create Hinglish keywords after Hindi is processed
        if 'hindi' in multilingual_db['languages'] and 'hinglish' not in multilingual_db['languages']:
            hindi_data = multilingual_db['languages']['hindi']
            hinglish_keywords = self.create_hinglish_keywords(english_db, hindi_data)
            multilingual_db['languages']['hinglish'] = hinglish_keywords
        
        # Finalize statistics
        for lang_name in self.all_languages.keys():
            if lang_name in multilingual_db['languages']:
                multilingual_db['languages'][lang_name]['keywords'] = list(
                    set(multilingual_db['languages'][lang_name]['keywords'])
                )
                multilingual_db['languages'][lang_name]['total_keywords'] = len(
                    multilingual_db['languages'][lang_name]['keywords']
                )
                
                # Calculate success rate for this language
                if lang_name not in ['english', 'hinglish']:
                    total_translations = total_words_processed
                    successful_for_lang = sum(1 for word_list in multilingual_db['languages'][lang_name]['categories'].values() 
                                            for word in word_list if word.get('translation_success', False))
                    
                    if total_translations > 0:
                        multilingual_db['languages'][lang_name]['translation_success_rate'] = (successful_for_lang / total_translations) * 100
                else:
                    # English and Hinglish have 100% success rate
                    multilingual_db['languages'][lang_name]['translation_success_rate'] = 100.0
        
        multilingual_db['metadata']['total_english_words'] = total_words_processed
        multilingual_db['metadata']['total_translations'] = total_words_processed * len(target_languages)
        multilingual_db['metadata']['successful_translations'] = successful_translations
        multilingual_db['metadata']['overall_success_rate'] = (successful_translations / (total_words_processed * len(target_languages))) * 100
        multilingual_db['metadata']['build_duration_seconds'] = time.time() - start_time
        multilingual_db['metadata']['failed_translations'] = self.failed_translations
        
        # Clean up checkpoint after successful completion
        self.cleanup_checkpoint()
        if os.path.exists('config/partial_multilingual_database.json'):
            os.remove('config/partial_multilingual_database.json')
        
        return multilingual_db
    
    def save_database(self, database):
        """Save the multilingual database to files"""
        try:
            # Create multilingual directory
            os.makedirs('config/multilingual_keywords', exist_ok=True)
            
            # Save complete database
            complete_filename = 'config/complete_multilingual_database.json'
            with open(complete_filename, 'w', encoding='utf-8') as f:
                json.dump(database, f, ensure_ascii=False, indent=2)
            self.logger.success(f"Complete database saved to: {complete_filename}")
            
            # Save individual language files
            for lang_name, lang_data in database['languages'].items():
                lang_filename = f'config/multilingual_keywords/keywords_{lang_name}.json'
                
                lang_file_data = {
                    'language': lang_name,
                    'total_keywords': lang_data['total_keywords'],
                    'translation_success_rate': lang_data.get('translation_success_rate', 0),
                    'keywords': lang_data['keywords'][:200],
                    'categories': lang_data['categories']
                }
                
                with open(lang_filename, 'w', encoding='utf-8') as f:
                    json.dump(lang_file_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"   💾 {lang_name.title():<10} {lang_data['total_keywords']:>4} keywords "
                               f"({lang_data.get('translation_success_rate', 0):.1f}% success)")
            
            # Save consolidated keywords for quick access
            consolidated = {}
            for lang_name, lang_data in database['languages'].items():
                consolidated[lang_name] = lang_data['keywords'][:100]
            
            consolidated_filename = 'config/consolidated_keywords.json'
            with open(consolidated_filename, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, ensure_ascii=False, indent=2)
            self.logger.success(f"Consolidated keywords saved to: {consolidated_filename}")
            
            # Save failed translations for review
            if self.failed_translations:
                failed_filename = 'config/failed_translations.json'
                with open(failed_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_translations, f, ensure_ascii=False, indent=2)
                self.logger.warning(f"Failed translations saved to: {failed_filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving database: {e}")
            return False
    
    def generate_stats_report(self, database):
        """Generate a statistics report"""
        if not database:
            return
        
        total_keywords_all_languages = sum(
            lang_data['total_keywords'] for lang_data in database['languages'].values()
        )
        
        self.logger.success("🎉 REAL MULTILINGUAL DATABASE BUILD COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info("📊 REAL TRANSLATION STATISTICS:")
        self.logger.info("=" * 60)
        
        for lang_name, lang_data in database['languages'].items():
            success_rate = lang_data.get('translation_success_rate', 0)
            self.logger.info(f"🌐 {lang_name.upper():<10} {lang_data['total_keywords']:>4} keywords "
                           f"| {success_rate:>5.1f}% success")
        
        self.logger.info("=" * 60)
        self.logger.info(f"📈 TOTAL KEYWORDS: {total_keywords_all_languages:,}")
        self.logger.info(f"✅ Overall success rate: {database['metadata']['overall_success_rate']:.1f}%")
        self.logger.info(f"⏱️  Build time: {database['metadata']['build_duration_seconds']:.1f} seconds")
        if database['metadata'].get('resumed_from_checkpoint'):
            self.logger.info("🔄 RESUMED FROM CHECKPOINT: Yes")
        self.logger.info("=" * 60)
        
        self.logger.info("💾 FILES CREATED:")
        self.logger.info("   📄 config/complete_multilingual_database.json")
        self.logger.info("   📁 config/multilingual_keywords/ (14 language files)")
        self.logger.info("   📄 config/consolidated_keywords.json")
        self.logger.info("=" * 60)
        self.logger.success("🚀 Ready for REAL Phase 1 Data Collection!")

def main():
    """Main execution function"""
    logger = setup_simple_logger()
    logger.info("🌍 BUILDING REAL MULTILINGUAL KEYWORD DATABASE")
    logger.info("   INCLUDES: English + Hinglish + 12 Indian Languages")
    logger.info("   Using Google Translate API")
    logger.info("=" * 60)
    
    # Check if deep-translator is installed
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        logger.error("❌ deep-translator not installed!")
        logger.info("💡 Run: pip install deep-translator")
        return
    
    try:
        # Initialize REAL translator
        translator = RealTranslator()
        
        # Build the database with REAL translations
        logger.info("Starting OPTIMIZED translation process...")
        multilingual_db = translator.build_multilingual_database()
        
        if not multilingual_db:
            logger.error("Failed to build multilingual database")
            return
        
        # Save the database
        if translator.save_database(multilingual_db):
            # Generate statistics
            translator.generate_stats_report(multilingual_db)
        else:
            logger.error("Failed to save database files")
            
    except KeyboardInterrupt:
        logger.info("Build process interrupted by user")
        logger.info("💾 Checkpoint saved! Run again to resume from where you left off.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()