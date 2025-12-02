# # Copyright (C) KonaAI - All Rights Reserved
"""TextProcessor class"""
import string

from nltk import pos_tag
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from src.utils.status import Status

# Eagerly load WordNet data, else it errors when multiple processes try to load it
_ = list(wordnet.all_synsets())


class TextProcessor:
    """TextProcessor class"""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        stopwords=None,
        remove_stopwords=True,
        remove_punctuation=True,
        remove_digits=True,
        remove_short_words=True,
        remove_non_ascii=True,
        lemmatize=True,
    ):
        """Init TextProcessor class

        Args:
            stopwords (list, optional): List of stopwards. Defaults to None.
            remove_stopwords (bool, optional): To remove stop words. Defaults to True.
            remove_punctuation (bool, optional): To remove puctuations. Defaults to True.
            remove_digits (bool, optional): To remove numbers. Defaults to True.
            remove_short_words (bool, optional): To remove words small than 3 letters. Defaults to True.
            remove_non_ascii (bool, optional): To remove non-ascii characters. Defaults to True.
            lemmatize (bool, optional): To lemmatize text. Defaults to True.
        """
        if stopwords is None:
            self.stopwords = set(nltk_stopwords.words("english"))
        else:
            self.stopwords = stopwords
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_short_words = remove_short_words
        self.remove_non_ascii = remove_non_ascii
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()

    def process_str(self, text: str):
        """_summary_: Process text"""
        try:
            if self.remove_digits:
                text = self.__remove_digits__(text)
            if self.remove_punctuation:
                text = self.__remove_punctuation__(text)
            if self.remove_short_words:
                text = self.__remove_short_words__(text)
            if self.remove_non_ascii:
                text = self.__remove_non_ascii__(text)
            if self.remove_stopwords:
                text = self.__remove_stop_words__(text)
            if self.lemmatize:
                text = self.__lemmatize_text__(text)
        except BaseException as _e:
            Status.FAILED("Error while processing text", error=str(_e))
        return text

    def process_df(self, _df, input_text_column: str, output_text_column: str):
        """Process text"""
        try:
            if self.remove_digits:
                _df[output_text_column] = _df[input_text_column].map(
                    self.__remove_digits__,
                    na_action="ignore",
                )
            if self.remove_punctuation:
                _df[output_text_column] = _df[output_text_column].map(
                    self.__remove_punctuation__,
                    na_action="ignore",
                )
            if self.remove_short_words:
                _df[output_text_column] = _df[output_text_column].map(
                    self.__remove_short_words__,
                    na_action="ignore",
                )
            if self.remove_non_ascii:
                _df[output_text_column] = _df[output_text_column].map(
                    self.__remove_non_ascii__,
                    na_action="ignore",
                )
            if self.remove_stopwords:
                _df[output_text_column] = _df[output_text_column].map(
                    self.__remove_stop_words__,
                    na_action="ignore",
                )
            if self.lemmatize:
                _df[output_text_column] = _df[output_text_column].map(
                    self.__lemmatize_text__,
                    na_action="ignore",
                )
        except BaseException as _e:
            Status.FAILED("Error while processing text", error=str(_e))

        return _df

    # remove all short words from text  (length < 3)
    def __remove_short_words__(self, text: str):
        return " ".join([word for word in text.split() if len(word) > 2])

    # remove all non-ascii characters
    def __remove_non_ascii__(self, text: str):
        return "".join([i if ord(i) < 128 else " " for i in text])

    # remove all punctuation
    def __remove_punctuation__(self, text: str):
        return text.translate(str.maketrans("", "", string.punctuation)) if text else ""

    # remove all digits
    def __remove_digits__(self, text: str):
        return text.translate(str.maketrans("", "", string.digits))

    # lemmatize text
    def __lemmatize_text__(self, text: str, lemmatizer=None):
        if lemmatizer is None:
            lemmatizer = self.lemmatizer
        lemmatized_text = []
        for word, tag in pos_tag(text.split()):
            if tag.startswith("NN"):
                pos = "n"
            elif tag.startswith("VB"):
                pos = "v"
            else:
                pos = "a"
            lemmatized_text.append(lemmatizer.lemmatize(word, pos))
        return " ".join(lemmatized_text)

    # remove all stop words from text
    def __remove_stop_words__(self, text: str, stop_words=None):
        if stop_words is None:
            stop_words = self.stopwords
        return " ".join([word for word in text.split() if word not in stop_words])
