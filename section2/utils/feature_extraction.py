# utils/feature_extraction.py

def extract_context_window(instance, window_size=10):
    """
    Extract the context words from left and right context within the specified window size.
    :param instance: A dictionary containing 'left_context', 'target', 'right_context'.
    :param window_size: Number of words to consider from both sides of the target word.
    :return: List of words in the context window.
    """
    left_context = instance['left_context'][-window_size:]  # Get the last `window_size` words from the left context
    right_context = instance['right_context'][:window_size]  # Get the first `window_size` words from the right context
    context_window = left_context + right_context
    return tuple(context_window)


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
    :return: Dictionary of word pair features as tuples.
    """
    features = {}
    if len(instance['left_context']) >= 2:
        features['word_pair_-2_-1'] = (instance['left_context'][-2], instance['left_context'][-1])  # Pair of words (-2, -1)
    if len(instance['left_context']) >= 1 and len(instance['right_context']) >= 1:
        features['word_pair_-1_+1'] = (instance['left_context'][-1], instance['right_context'][0])  # Pair (-1, +1)
    if len(instance['right_context']) >= 2:
        features['word_pair_+1_+2'] = (instance['right_context'][0], instance['right_context'][1])  # Pair (+1, +2)
    
    # Convert pairs to tuples (they are already tuples by nature, but being explicit here)
    for key, value in features.items():
        features[key] = tuple(value)
    
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