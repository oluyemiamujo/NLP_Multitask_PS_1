def build_decision_list(ranked_features, sense_label1, sense_label2):
    """
    Build the decision list using the ranked features and their log-likelihood scores.
    Each decision rule will predict one of the two senses (sense_label1 or sense_label2).
    
    :param ranked_features: List of tuples (feature, log-likelihood score) sorted by score.
    :param sense_label1: The label for the first sense (e.g., "*bass").
    :param sense_label2: The label for the second sense (e.g., "bass").
    :return: A decision list as a list of dictionaries, each containing 'feature', 'sense', and 'score'.
    """
    decision_list = []

    for feature, score in ranked_features:
        # If the log-likelihood score is positive, predict sense_label1; if negative, predict sense_label2
        if score > 0:
            predicted_sense = sense_label1
        else:
            predicted_sense = sense_label2

        decision_list.append({
            'feature': feature,
            'sense': predicted_sense,
            'score': score
        })

    return decision_list

def classify_instance_with_decision_list(instance, decision_list):
    """
    Classify a test instance using the decision list. The classification is based on the first matching feature.
    
    :param instance: A dictionary representing the test instance, containing features and the target word.
    :param decision_list: The decision list constructed from the ranked features.
    :return: The predicted sense of the test instance.
    """
    for rule in decision_list:
        feature, predicted_sense = rule['feature'], rule['sense']
        
        # Check if the feature in the decision list matches the instance
        if feature in instance.items():
            return predicted_sense

    # If no feature from the decision list matches, return None (we'll handle this later with a default sense)
    return None


def classify_test_data(test_data, decision_list, default_sense):
    """
    Classify all test instances using the decision list.
    
    :param test_data: List of test instances.
    :param decision_list: The decision list used to classify the instances.
    :param default_sense: The default sense to use if no feature matches.
    :return: List of predicted senses.
    """
    predictions = []
    
    for instance in test_data:
        # Classify each instance using the decision list
        predicted_sense = classify_instance_with_decision_list(instance, decision_list)
        
        # If no feature matches, use the default sense
        if predicted_sense is None:
            predicted_sense = default_sense
        
        predictions.append(predicted_sense)
    
    return predictions

def get_default_sense(train_data, sense_label1, sense_label2):
    """
    Determine the most common sense in the training data to use as the default classification.
    
    :param train_data: The training data containing sense labels.
    :param sense_label1: The first sense (e.g., "*bass").
    :param sense_label2: The second sense (e.g., "bass").
    :return: The most common sense in the training data.
    """
    count_sense1 = sum(1 for instance in train_data if instance['sense'] == sense_label1)
    count_sense2 = sum(1 for instance in train_data if instance['sense'] == sense_label2)
    
    # Return the more frequent sense
    if count_sense1 >= count_sense2:
        return sense_label1
    else:
        return sense_label2



