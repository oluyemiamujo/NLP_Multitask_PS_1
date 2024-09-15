# utils/feature_extraction.py

def extract_context_window(instance, window_size=10):
    """
    Extract the context words from left and right context within the specified window size.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :param window_size: Number of words to consider from both sides of the target word.
    :return: Tuple of words in the context window.
    """
    left_context = instance['left_context'][-window_size:]  # Get the last `window_size` words from the left context
    right_context = instance['right_context'][:window_size]  # Get the first `window_size` words from the right context
    context_window = left_context + right_context
    return tuple(context_window)  # Convert to tuple


def extract_position_based_features(instance):
    """
    Extract features based on specific positions relative to the target word.
    Example: Word at position -1, word at position +1, etc.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :return: Dictionary of position-based features.
    """
    features = {}
    if len(instance['left_context']) >= 1:
        features['word_-1'] = instance['left_context'][-1]  # Word immediately to the left (-1)
    if len(instance['right_context']) >= 1:
        features['word_+1'] = instance['right_context'][0]  # Word immediately to the right (+1)
    
    if len(instance['left_context']) >= 2:
        features['word_-2'] = instance['left_context'][-2]  # Word two positions to the left (-2)
    if len(instance['right_context']) >= 2:
        features['word_+2'] = instance['right_context'][1]  # Word two positions to the right (+2)
    
    return features


def extract_word_pairs(instance):
    """
    Extract word pairs from the left and right context.
    Example: Pair (-1, +1), (-2, -1), etc.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :return: Dictionary of word pair features.
    """
    features = {}
    if len(instance['left_context']) >= 2:
        features['word_pair_-2_-1'] = (instance['left_context'][-2], instance['left_context'][-1])  # Pair of words (-2, -1)
    if len(instance['left_context']) >= 1 and len(instance['right_context']) >= 1:
        features['word_pair_-1_+1'] = (instance['left_context'][-1], instance['right_context'][0])  # Pair (-1, +1)
    if len(instance['right_context']) >= 2:
        features['word_pair_+1_+2'] = (instance['right_context'][0], instance['right_context'][1])  # Pair (+1, +2)
    
    return features


def extract_bag_of_words(instance):
    """
    Create a bag-of-words feature from the context window.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :return: Dictionary with a single feature 'bag_of_words' containing a tuple of context words.
    """
    context_window = extract_context_window(instance)
    return {'bag_of_words': tuple(sorted(context_window))}  # Convert set to sorted tuple


def extract_all_features(instance):
    """
    Extract all features for a given instance. This is the main function that will be used in the training/testing pipeline.
    It combines context window, position-based, word pairs, and bag-of-words features.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :return: Dictionary containing all extracted features.
    """
    features = {}
    
    # Extract context window
    context_window = extract_context_window(instance)
    features['context_window'] = context_window
    
    # Extract position-based features
    position_features = extract_position_based_features(instance)
    features.update(position_features)
    
    # Extract word pairs
    word_pairs = extract_word_pairs(instance)
    features.update(word_pairs)
    
    # Extract bag of words
    bow_feature = extract_bag_of_words(instance)
    features.update(bow_feature)
    
    return features

# import nltk

# from functions.preprocessing import remove_accents
# nltk.download('averaged_perceptron_tagger')

# # Extract context features (±window_size words) around the ambiguous word
# def extract_features(corpus, ambiguity_dict, window_size=10):
#     """
#     Extracts context features (±window_size words) for ambiguous words in the corpus.
#     """
#     features = []
#     for idx, word in enumerate(corpus):
#         base_word = remove_accents(word)
#         if base_word in ambiguity_dict:
#             # Get context around the ambiguous word
#             context_before = corpus[max(0, idx - window_size): idx]
#             context_after = corpus[idx + 1: min(len(corpus), idx + window_size + 1)]
#             features.append((base_word, word, context_before + context_after))
#     return features

# # Extract words in position -k and +k relative to the ambiguous word
# def extract_position_features(corpus, ambiguity_dict, k):
#     """
#     Extracts words in position -k and +k relative to the ambiguous word.
#     """
#     position_features = []
#     for idx, word in enumerate(corpus):
#         base_word = remove_accents(word)
#         if base_word in ambiguity_dict:
#             before_k = corpus[max(0, idx - k)] if idx - k >= 0 else None
#             after_k = corpus[idx + k] if idx + k < len(corpus) else None
#             position_features.append((base_word, word, before_k, after_k))
#     return position_features

# # Extract POS tags for context words around ambiguous words
# def extract_pos_features(corpus, ambiguity_dict, window_size=10):
#     """
#     Extracts POS tags for context words around ambiguous words.
#     """
#     pos_features = []
#     for idx, word in enumerate(corpus):
#         base_word = remove_accents(word)
#         if base_word in ambiguity_dict:
#             context_before = corpus[max(0, idx - window_size): idx]
#             context_after = corpus[idx + 1: min(len(corpus), idx + window_size + 1)]
#             context = context_before + context_after
#             # Get POS tags for the context words
#             pos_tags = nltk.pos_tag(context)
#             pos_features.append((base_word, word, pos_tags))
#     return pos_features

# # Extract POS tags for words in position -k and +k relative to the ambiguous word
# def extract_pos_position_features(corpus, ambiguity_dict, k):
#     """
#     Extracts POS tags for words in position -k and +k relative to the ambiguous word.
#     """
#     pos_position_features = []
#     for idx, word in enumerate(corpus):
#         base_word = remove_accents(word)
#         if base_word in ambiguity_dict:
#             before_k = corpus[max(0, idx - k)] if idx - k >= 0 else None
#             after_k = corpus[idx + k] if idx + k < len(corpus) else None
#             if before_k:
#                 before_k_pos = nltk.pos_tag([before_k])[0][1]  # Get POS for word in position -k
#             else:
#                 before_k_pos = None
#             if after_k:
#                 after_k_pos = nltk.pos_tag([after_k])[0][1]  # Get POS for word in position +k
#             else:
#                 after_k_pos = None
#             pos_position_features.append((base_word, word, before_k_pos, after_k_pos))
#     return pos_position_features