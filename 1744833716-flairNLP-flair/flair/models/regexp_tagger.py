import re
from dataclasses import dataclass, field
from typing import Union

from flair.data import Sentence, Span, Token


@dataclass
class TokenCollection:
    """A utility class for RegexpTagger to hold all tokens for a given Sentence and define some functionality.

    Args:
        sentence: A Sentence object
    """

    sentence: Sentence
    __tokens_start_pos: list[int] = field(init=False, default_factory=list)
    __tokens_end_pos: list[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        for token in self.tokens:
            self.__tokens_start_pos.append(token.start_position)
            self.__tokens_end_pos.append(token.end_position)

    @property
    def tokens(self) -> list[Token]:
        return list(self.sentence)

    def get_token_span(self, span: tuple[int, int]) -> Span:
        """Find a span by the token character positions.

        Given an interval specified with start and end pos as tuple, this function returns a Span object
        spanning the tokens included in the interval. If the interval is overlapping with a token span, a
        ValueError is raised

        Args:
            span: Start and end pos of the requested span as tuple

        Returns: A span object spanning the requested token interval
        """
        span_start: int = self.__tokens_start_pos.index(span[0])
        span_end: int = self.__tokens_end_pos.index(span[1])
        return Span(self.tokens[span_start : span_end + 1])


class RegexpTagger:
    def __init__(
        self, mapping: Union[list[Union[tuple[str, str], tuple[str, str, int]]], tuple[str, str], tuple[str, str, int]]
    ) -> None:
        r"""This tagger is capable of tagging sentence objects with given regexp -> label mappings.

        I.e: The tuple (r'(["\'])(?:(?=(\\?))\2.)*?\1', 'QUOTE') maps every match of the regexp to
        a <QUOTE> labeled span and therefore labels the given sentence object with RegexpTagger.predict().
        This tagger supports multilabeling so tokens can be included in multiple labeled spans.
        The regexp are compiled internally and an re.error will be raised if the compilation of a given regexp fails.

        If a match violates (in this case overlaps) a token span, an exception is raised.

        Args:
            mapping: A list of tuples or a single tuple representing a mapping as regexp -> label
        """
        self._regexp_mapping: list = []
        self.register_labels(mapping=mapping)

    def label_type(self):
        for regexp, label, group in self._regexp_mapping:
            return label

    @property
    def registered_labels(self):
        return self._regexp_mapping

    def register_labels(self, mapping):
        """Register a regexp -> label mapping.

        Args:
            mapping: A list of tuples or a single tuple representing a mapping as regexp -> label
        """
        mapping = self._listify(mapping)

        for entry in mapping:
            regexp = entry[0]
            label = entry[1]
            group = entry[2] if len(entry) > 2 else 0
            try:
                pattern = re.compile(regexp)
                self._regexp_mapping.append((pattern, label, group))

            except re.error as err:
                raise re.error(
                    f"Couldn't compile regexp '{regexp}' for label '{label}'. Aborted with error: '{err.msg}'"
                )

    def remove_labels(self, labels: Union[list[str], str]):
        """Remove a registered regexp -> label mapping given by label.

        Args:
            labels: A list of labels or a single label as strings.
        """
        labels = self._listify(labels)

        self._regexp_mapping = [mapping for mapping in self._regexp_mapping if mapping[1] not in labels]

    @staticmethod
    def _listify(element: object) -> list:
        if not isinstance(element, list):
            return [element]
        else:
            return element

    def predict(self, sentences: Union[list[Sentence], Sentence]) -> list[Sentence]:
        """Predict the given sentences according to the registered mappings."""
        if not isinstance(sentences, list):
            sentences = [sentences]
        if not sentences:
            return sentences

        sentences = self._listify(sentences)
        for sentence in sentences:
            self._label(sentence)
        return sentences

    def _label(self, sentence: Sentence):
        """This will add a complex_label to the given sentence for every match.span() for every registered_mapping.

        If a match span overlaps with a token span an exception is raised.
        """
        collection = TokenCollection(sentence)

        for pattern, label, group in self._regexp_mapping:
            for match in pattern.finditer(sentence.to_original_text()):
                # print(match)
                span: tuple[int, int] = match.span(group)
                # print(span)
                try:
                    token_span = collection.get_token_span(span)
                except ValueError:
                    raise Exception(f"The match span {span} for label '{label}' is overlapping with a token!")
                token_span.add_label(label, label)
