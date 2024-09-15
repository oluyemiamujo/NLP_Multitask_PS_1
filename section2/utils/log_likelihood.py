# utils/log_likelihood.py

import math
from collections import defaultdict

def count_feature_frequencies(feature_data, sense_label1, sense_label2):
    """
    Count the occurrences of each feature for each sense (sense_label1 and sense_label2).
    :param feature_data: List of feature dictionaries from training data, each containing features and the 'sense' label.
    :param sense_label1: The first sense (e.g., "*bass" for fish).
    :param sense_label2: The second sense (e.g., "bass" for music).
    :return: Two dictionaries with feature counts for each sense.
    """
    freq_sense1 = defaultdict(int)  # Feature counts for sense_label1
    freq_sense2 = defaultdict(int)  # Feature counts for sense_label2
    total_sense1 = 0  # Total count for sense_label1
    total_sense2 = 0  # Total count for sense_label2
    
    # Debugging: Add print statements
    print(f"Counting features for sense: {sense_label1} and {sense_label2}")
    
    for instance in feature_data:
        sense = instance['sense']
        features = instance  # All other keys are features
        
        if sense == sense_label1:
            total_sense1 += 1
            for feature, value in features.items():
                if feature != 'sense':
                    freq_sense1[(feature, value)] += 1
                    print(f"Sense: {sense_label1}, Feature: {feature}, Value: {value}, Count: {freq_sense1[(feature, value)]}")
        
        elif sense == sense_label2:
            total_sense2 += 1
            for feature, value in features.items():
                if feature != 'sense':
                    freq_sense2[(feature, value)] += 1
                    print(f"Sense: {sense_label2}, Feature: {feature}, Value: {value}, Count: {freq_sense2[(feature, value)]}")
    
    return freq_sense1, freq_sense2, total_sense1, total_sense2

def calculate_probabilities(feature_counts, total_sense, smoothing_factor=1, num_unique_features=1):
    """
    Calculate probabilities for each feature given the sense.
    Apply Laplace smoothing.
    :param feature_counts: Dictionary of feature counts for a particular sense.
    :param total_sense: Total number of instances for that sense.
    :param smoothing_factor: Value to apply for Laplace smoothing.
    :param num_unique_features: Total number of unique features.
    :return: A dictionary with probabilities for each feature.
    """
    probabilities = {}
    
    for feature, count in feature_counts.items():
        probabilities[feature] = (count + smoothing_factor) / (total_sense + smoothing_factor * num_unique_features)
    
    return probabilities


def calculate_log_likelihood(prob_sense1, prob_sense2):
    """
    Calculate log-likelihood ratios for each feature.
    :param prob_sense1: Dictionary of probabilities for each feature for sense_label1.
    :param prob_sense2: Dictionary of probabilities for each feature for sense_label2.
    :return: A dictionary mapping features to their log-likelihood ratios.
    """
    log_likelihoods = {}
    
    for feature in prob_sense1:
        prob_1 = prob_sense1.get(feature, 0)  # Probability for sense_label1
        prob_2 = prob_sense2.get(feature, 0)  # Probability for sense_label2
        
        # Avoid division by zero
        if prob_2 == 0:
            prob_2 = 1e-6
        
        # Calculate log-likelihood ratio
        log_likelihoods[feature] = math.log(prob_1 / prob_2)
    
    return log_likelihoods


def rank_features_by_log_likelihood(log_likelihoods):
    """
    Rank features based on their log-likelihood values.
    :param log_likelihoods: Dictionary mapping features to log-likelihood values.
    :return: A list of features sorted by their log-likelihood values (in descending order).
    """
    ranked_features = sorted(log_likelihoods.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked_features