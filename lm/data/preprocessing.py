from __future__ import annotations

import abc
import re
import unicodedata


class TextSamplePreprocessor(abc.ABC):
    def __call__(self, text: str) -> str:
        return self.preprocess(text)

    @abc.abstractmethod
    def preprocess(self, text: str) -> str:
        pass


class ConversationPreprocessor(TextSamplePreprocessor):
    EMOJI_REPLACEMENTS = {
        ":-D": ":D",
        ":-)": ":)",
        ":-(": ":(",
        ":-P": ":P",
        ":-*": ":*"
    }
    EMOJIS_PATTERN = re.compile("|".join(re.escape(key) for key in EMOJI_REPLACEMENTS))
    URL_PATTERN = re.compile(r"http\S+")
    WHITESPACE_PATTERN = re.compile(r"\s+", re.M)
    LEADING_DASH_PATTERN = re.compile(r"^\s*-\s*")

    def preprocess(self, text: str) -> str:
        text = text.strip()
        text = self.remove_urls(text)
        text = self.remove_accents(text)
        text = self.normalize_whitespaces(text)
        text = self.normalize_emojis(text)
        text = self.remove_leading_dash(text)
        text = self.truncate_strings_shorter_than(text, min_length=10)
        return text

    @staticmethod
    def remove_accents(text: str) -> str:
        nfkd_form = unicodedata.normalize("NFKD", text)
        only_ascii = nfkd_form.encode("ASCII", "ignore").decode("utf-8")
        return only_ascii

    @classmethod
    def remove_urls(cls, text: str) -> str:
        return cls.URL_PATTERN.sub("", text)

    @classmethod
    def normalize_whitespaces(cls, text: str) -> str:
        return cls.WHITESPACE_PATTERN.sub(" ", text)

    @classmethod
    def normalize_emojis(cls, text: str) -> str:
        return cls.EMOJIS_PATTERN.sub(lambda match: cls.EMOJI_REPLACEMENTS[match.group(0)], text)
    
    @classmethod
    def remove_leading_dash(cls, text: str) -> str:
        return cls.LEADING_DASH_PATTERN.sub("", text)
    
    @staticmethod
    def truncate_strings_shorter_than(text: str, *, min_length: int = 1) -> str:
        return "" if len(text) < min_length else text
