from sklearn.metrics import accuracy_score, confusion_matrix

# def evaluate_classifier(predictions, true_labels):
#     """
#     Evaluate the classifier by computing accuracy and confusion matrix.
    
#     :param predictions: List of predicted senses from the classifier.
#     :param true_labels: List of actual senses (ground truth).
#     :return: Accuracy and confusion matrix.
#     """
#     # Calculate accuracy
#     accuracy = accuracy_score(true_labels, predictions)
    
#     # Create confusion matrix
#     conf_matrix = confusion_matrix(true_labels, predictions, labels=list(set(true_labels)))
    
#     return accuracy, conf_matrix

from sklearn.metrics import precision_recall_fscore_support

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

