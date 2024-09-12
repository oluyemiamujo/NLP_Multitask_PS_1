import re
from collections import defaultdict

# Convert text to lowercase and remove punctuation
def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase and removing punctuation.
    Keeps only letters and spaces.
    """
    text = text.lower()
    # Keep letters and accented characters
    text = re.sub(r'[^a-záéíóúàèìòùñ\s]', '', text)
    return text.split()

# Remove accents from words
def remove_accents(word):
    """
    Removes accents from the input word.
    """
    accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
               'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u', 'ñ': 'n'}
    return ''.join([accents.get(c, c) for c in word])

# Identify ambiguous words based on accents
def identify_ambiguous_words(corpus):
    """
    Identifies words that have multiple accented forms.
    """
    ambiguity_dict = defaultdict(set)
    for word in corpus:
        base_word = remove_accents(word)
        ambiguity_dict[base_word].add(word)
    
    # Only return words with multiple forms
    ambiguous_words = {key: val for key, val in ambiguity_dict.items() if len(val) > 1}
    return ambiguous_words