import math
from collections import defaultdict

# Measure collocational frequencies
def measure_collocations(training_data):
    """
    Measures frequency of collocations (context word + ambiguous word).
    """
    collocation_freq = defaultdict(lambda: defaultdict(int))
    for base_word, accented_word, context in training_data:
        for context_word in context:
            collocation_freq[base_word][(context_word, accented_word)] += 1
    return collocation_freq

# Calculate log-likelihood for collocations
def calculate_log_likelihood(collocation_freq, total_collocations):
    """
    Calculates log-likelihood for collocations to rank their reliability for disambiguation.
    """
    log_likelihoods = []
    for base_word, collocations in collocation_freq.items():
        for (context_word, accented_word), observed_freq in collocations.items():
            expected_freq = total_collocations.get(context_word, 1)  # Avoid division by zero
            log_likelihood = math.log(observed_freq / expected_freq)
            log_likelihoods.append((base_word, (context_word, accented_word), log_likelihood))
    return log_likelihoods

# Build the decision list based on log-likelihoods
def build_decision_list(log_likelihoods):
    """
    Builds a decision list by ranking collocations based on log-likelihood.
    """
    # Sort by log-likelihood in descending order
    decision_list = sorted(log_likelihoods, key=lambda x: -x[2])
    return decision_list

# Use the decision list to classify a new ambiguous word based on its context
def classify_word(context, decision_list):
    """
    Classifies ambiguous words using the decision list.
    Looks for the most reliable feature in the context and selects the corresponding form.
    """
    for base_word, (context_word, accented_word), log_likelihood in decision_list:
        if context_word in context:
            return accented_word  # Return the most likely form
    return None  # Return None if no match is found