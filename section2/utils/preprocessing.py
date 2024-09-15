import re
import string

def load_data(file_path):
    """
    Reads the file, processes it line by line and returns a structured data format.
    :param file_path: path to the data file.
    :return: List of dictionaries containing context, target word, and sense.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            processed_instance = process_line(line.strip())
            if processed_instance:
                data.append(processed_instance)
    return data


def process_line(line):
    """
    Processes a single line of the input data to extract the sense, context, and target word.
    :param line: A string representing one line from the file.
    :return: A dictionary with 'sense', 'left_context', 'target', and 'right_context'
    """
    # Match the format (*bass, bass) : five words of left context, target word, five words of right context
    pattern = r"^(\*?bass|\*?sake):(.+)"
    match = re.match(pattern, line)
    
    if match:
        sense = match.group(1).strip()
        context = match.group(2).strip()
        left_context, target, right_context = extract_context(context)
        return {
            'sense': sense,
            'left_context': left_context,
            'target': target,
            'right_context': right_context
        }
    return None


def extract_context(context):
    """
    Splits the context into left, target word, and right context. Assumes the target word is in the middle.
    :param context: String containing 11 words (5 left, 1 target, 5 right).
    :return: left_context, target, right_context
    """
    words = context.split()
    if len(words) != 11:
        raise ValueError(f"Expected 11 words in the context, got {len(words)}.")
    
    left_context = words[:5]
    target = words[5]
    right_context = words[6:]
    
    return left_context, target, right_context


def clean_text(text):
    """
    Converts text to lowercase, removes punctuation but keeps accents.
    :param text: Input string
    :return: Cleaned string
    """
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text