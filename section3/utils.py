import re
import nltk
import random
from nltk.util import bigrams, trigrams
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to remove extraneous content from Gutenberg texts
def remove_gutenberg_boilerplate(text):
    start_marker = r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*'
    end_marker = r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*'
    text = re.split(start_marker, text, flags=re.IGNORECASE)[-1]
    text = re.split(end_marker, text, flags=re.IGNORECASE)[0]
    return text.strip()

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to tokenize the cleaned text
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Full preprocessing pipeline for Gutenberg texts
def preprocess_gutenberg_text(raw_text):
    clean_raw_text = remove_gutenberg_boilerplate(raw_text)
    cleaned_text = clean_text(clean_raw_text)
    tokens = tokenize_text(cleaned_text)
    return tokens

# Function to generate text using word bigrams
def generate_bigram_text(start_word, bigram_freq, num_words=50):
    """
    Generates text using the bigram model.
    Starts with a given word and generates the next word based on bigram frequencies.
    """
    text = [start_word]
    current_word = start_word

    for _ in range(num_words - 1):
        if current_word in bigram_freq:
            next_word = random.choices(list(bigram_freq[current_word].keys()), 
                                       weights=bigram_freq[current_word].values())[0]
            text.append(next_word)
            current_word = next_word
        else:
            break

    return ' '.join(text)

# Function to generate text using character bigrams
def generate_char_bigram_text(start_char, char_bigram_freq, num_chars=100):
    """
    Generates text character by character using the character bigram model.
    """
    text = [start_char]
    current_char = start_char

    for _ in range(num_chars - 1):
        if current_char in char_bigram_freq:
            next_char = random.choices(list(char_bigram_freq[current_char].keys()), 
                                       weights=char_bigram_freq[current_char].values())[0]
            text.append(next_char)
            current_char = next_char
        else:
            break

    return ''.join(text)

# Function to generate text using word trigrams
def generate_trigram_text(start_bigram, trigram_freq, num_words=50):
    """
    Generates text using the trigram model (word-based).
    Starts with a bigram and generates subsequent words based on the trigram frequencies.
    """
    text = list(start_bigram)  # Initialize the generated text with the starting bigram
    current_bigram = start_bigram

    for _ in range(num_words - 2):
        if current_bigram in trigram_freq:
            next_word = random.choices(list(trigram_freq[current_bigram].keys()), 
                                       weights=trigram_freq[current_bigram].values())[0]
            text.append(next_word)
            current_bigram = (current_bigram[1], next_word)
        else:
            break

    return ' '.join(text)

# Function to identify content and function words
def get_content_and_function_words(tokens):
    """
    Separates content words and function words from the tokenized text.
    Function words are filtered based on NLTK's stopword list.
    """
    stop_words = set(stopwords.words('english'))

    # Perform frequency distribution on the tokens
    word_freq = FreqDist(tokens)

    # Separate function and content words
    function_words = {word: freq for word, freq in word_freq.items() if word in stop_words}
    content_words = {word: freq for word, freq in word_freq.items() if word not in stop_words}

    return function_words, content_words