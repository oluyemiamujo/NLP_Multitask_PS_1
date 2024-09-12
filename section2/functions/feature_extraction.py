import nltk
nltk.download('averaged_perceptron_tagger')

# Extract context features (±window_size words) around the ambiguous word
def extract_features(corpus, ambiguity_dict, window_size=10):
    """
    Extracts context features (±window_size words) for ambiguous words in the corpus.
    """
    features = []
    for idx, word in enumerate(corpus):
        base_word = remove_accents(word)
        if base_word in ambiguity_dict:
            # Get context around the ambiguous word
            context_before = corpus[max(0, idx - window_size): idx]
            context_after = corpus[idx + 1: min(len(corpus), idx + window_size + 1)]
            features.append((base_word, word, context_before + context_after))
    return features

# Extract words in position -k and +k relative to the ambiguous word
def extract_position_features(corpus, ambiguity_dict, k):
    """
    Extracts words in position -k and +k relative to the ambiguous word.
    """
    position_features = []
    for idx, word in enumerate(corpus):
        base_word = remove_accents(word)
        if base_word in ambiguity_dict:
            before_k = corpus[max(0, idx - k)] if idx - k >= 0 else None
            after_k = corpus[idx + k] if idx + k < len(corpus) else None
            position_features.append((base_word, word, before_k, after_k))
    return position_features

# Extract POS tags for context words around ambiguous words
def extract_pos_features(corpus, ambiguity_dict, window_size=10):
    """
    Extracts POS tags for context words around ambiguous words.
    """
    pos_features = []
    for idx, word in enumerate(corpus):
        base_word = remove_accents(word)
        if base_word in ambiguity_dict:
            context_before = corpus[max(0, idx - window_size): idx]
            context_after = corpus[idx + 1: min(len(corpus), idx + window_size + 1)]
            context = context_before + context_after
            # Get POS tags for the context words
            pos_tags = nltk.pos_tag(context)
            pos_features.append((base_word, word, pos_tags))
    return pos_features

# Extract POS tags for words in position -k and +k relative to the ambiguous word
def extract_pos_position_features(corpus, ambiguity_dict, k):
    """
    Extracts POS tags for words in position -k and +k relative to the ambiguous word.
    """
    pos_position_features = []
    for idx, word in enumerate(corpus):
        base_word = remove_accents(word)
        if base_word in ambiguity_dict:
            before_k = corpus[max(0, idx - k)] if idx - k >= 0 else None
            after_k = corpus[idx + k] if idx + k < len(corpus) else None
            if before_k:
                before_k_pos = nltk.pos_tag([before_k])[0][1]  # Get POS for word in position -k
            else:
                before_k_pos = None
            if after_k:
                after_k_pos = nltk.pos_tag([after_k])[0][1]  # Get POS for word in position +k
            else:
                after_k_pos = None
            pos_position_features.append((base_word, word, before_k_pos, after_k_pos))
    return pos_position_features