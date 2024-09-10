import re
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')

def remove_gutenberg_boilerplate(text):
    """
    Remove the extraneous content from Gutenberg texts such as headers, footers, 
    and any non-relevant content like chapter markers.
    """
    # Define markers for start and end of the actual book content
    start_marker = r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*'
    end_marker = r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*'

    # Remove everything before the START marker
    text = re.split(start_marker, text, flags=re.IGNORECASE)[-1]
    
    # Remove everything after the END marker
    text = re.split(end_marker, text, flags=re.IGNORECASE)[0]
    
    return text.strip()

def clean_text(text):
    """
    Perform general text cleaning:
    - Lowercase conversion
    - Remove non-alphabetical characters (punctuation, special characters)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove all non-alphabetical characters and keep only words and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    return text

def tokenize_text(text):
    """
    Tokenize the cleaned text using NLTK's word_tokenize.
    """
    # Tokenize the text using nltk
    return nltk.word_tokenize(text)

def preprocess_gutenberg_text(raw_text):
    """
    Full preprocessing pipeline for Gutenberg texts:
    - Remove headers/footers
    - Clean the text (lowercase, remove special characters)
    - Tokenize the text into words
    """
    # Step 1: Remove Gutenberg boilerplate
    clean_raw_text = remove_gutenberg_boilerplate(raw_text)
    
    # Step 2: Clean the text (lowercase and remove non-alphabetical characters)
    cleaned_text = clean_text(clean_raw_text)
    
    # Step 3: Tokenize the cleaned text
    tokens = tokenize_text(cleaned_text)
    
    return tokens