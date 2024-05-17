
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'labels'


def transformed_name(key):
    return key + '_xf'



stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
             "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
             "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
             "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
             "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
             "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def preprocessing_fn(inputs):
    outputs = {}

    # Text preprocessing
    text = tf.strings.lower(inputs['text'])
    text = tf.strings.regex_replace(text, r"(?:<br />)", " ")
    text = tf.strings.regex_replace(text, "n\'t", " not ")
    text = tf.strings.regex_replace(text, r"(?:\'ll|\'re|\'d|\'ve)", " ")
    text = tf.strings.regex_replace(text, r"\W+", " ")
    text = tf.strings.regex_replace(text, r"\d+", " ")
    text = tf.strings.regex_replace(text, r"\b[a-zA-Z]\b", " ")
    text = tf.strings.regex_replace(text, r'\b(' + r'|'.join(stopwords) + r')\b\s*', " ")
    text = tf.strings.strip(text)  # Remove leading and trailing whitespace

    outputs[transformed_name('text')] = text
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
