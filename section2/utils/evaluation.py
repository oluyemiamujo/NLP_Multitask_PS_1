from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from utils.decision_list import classify_test_instance

def evaluate_classifier_with_metrics(predictions, true_labels):
    """
    Evaluate the classifier by computing accuracy, confusion matrix, precision, recall, and F1-score.
    
    :param predictions: List of predicted senses from the classifier.
    :param true_labels: List of actual senses (ground truth).
    :return: Accuracy, confusion matrix, precision, recall, F1-score.
    """
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions, labels=list(set(true_labels)))
    
    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    
    return accuracy, conf_matrix, precision, recall, f1

def calculate_baseline_accuracy(test_data, most_frequent_sense):
    """
    Calculate baseline accuracy by predicting the most frequent sense for all test instances.
    
    :param test_data: List of dictionaries containing test data (with 'sense' key).
    :param most_frequent_sense: The most frequent sense in the training data.
    :return: Baseline accuracy as a percentage.
    """
    correct_predictions = sum(1 for instance in test_data if instance['sense'] == most_frequent_sense)
    total_instances = len(test_data)
    return correct_predictions / total_instances * 100  # Convert to percentage


def calculate_model_accuracy(test_data, decision_list, default_sense):
    """
    Calculate model accuracy using the decision list classifier.
    
    :param test_data: List of dictionaries containing test data (with 'sense' key).
    :param decision_list: The decision list containing ranked features and their corresponding senses.
    :param default_sense: The default sense to use if no feature in the decision list matches.
    :return: Model accuracy as a percentage.
    """
    correct_predictions = 0
    for instance in test_data:
        predicted_sense = classify_test_instance(instance, decision_list, default_sense)
        if predicted_sense == instance['sense']:
            correct_predictions += 1
    total_instances = len(test_data)
    return correct_predictions / total_instances * 100  # Convert to percentage

