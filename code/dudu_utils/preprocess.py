# Built-in
import string

# Libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Custom

# Settings


def lower_text(text_list):
    """
    Translate all text into lower case
    :param text_list: list of text
    :return: lower case list of text
    """
    return [a.lower() for a in text_list]


def remove_punctuation(text_list):
    """
    Remove punctuation
    :param text_list: list of text
    :return: punctuation removed list of text
    """
    table = str.maketrans('', '', string.punctuation.replace('\'', ''))
    return [a.translate(table) for a in text_list]


def tokenize(text_list):
    """
    Tokenize the sentence
    :param text_list: list of text
    :return: list of list, where each element in the sublist is the tokenized words
    """
    return [word_tokenize(a) for a in text_list]


def remove_stop_words(text_list):
    """
    Remove stop words
    :param text_list: list of list processed by tokenize()
    :return: stop words removed list of list
    """
    stop_words = stopwords.words('english')
    return [[b for b in a if b not in stop_words] for a in text_list]


def stemming(text_list):
    """
    Stemming the words using Porter stemmer
    :param text_list: list of list processed by tokenize()
    :return: stemmed list of list
    """
    porter = PorterStemmer()
    return [[porter.stem(b) for b in a] for a in text_list]


def make_preprocessor(preprocess_names, val):
    preproc_lut = {
        'lower_text': lower_text,
        'remove_punctuation': remove_punctuation,
        'tokenize': tokenize,
        'remove_stop_words': remove_stop_words,
        'stemming': stemming
    }
    for pn in preprocess_names:
        val = preproc_lut[pn](val)
    return val


if __name__ == '__main__':
    print(string.punctuation.replace('\'', ''))
