from __future__ import annotations

import abc
import json
import pathlib
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, Any, Iterable, Optional, ClassVar, Type


class InvalidTextSampleFile(Exception):
    pass


class TextSampleReader(abc.ABC):
    FILE_NAME_PATTERN: ClassVar[Type[str]] = NotImplemented

    def __init__(self, text_samples_file_paths: Iterable[str]) -> None:
        self.text_samples_file_paths = tuple(text_samples_file_paths)

    def __call__(self) -> Iterator[str]:
        return self.read_samples()

    @classmethod
    def from_directories(cls, root_dir_path: str, *args, **kwargs) -> TextSampleReader:
        text_samples_file_paths = (
            str(file) for file in pathlib.Path(root_dir_path).rglob(cls.FILE_NAME_PATTERN)
        )
        return cls(text_samples_file_paths, *args, **kwargs)

    def load(self) -> None:
        pass

    def read_samples(self) -> Iterator[str]:
        self.load()
        return self.read_samples_after_load()

    @abc.abstractmethod
    def read_samples_after_load(self) -> Iterator[str]:
        pass


class FacebookMessagesReader(TextSampleReader):
    FILE_NAME_PATTERN = "*.json"

    def __init__(
        self, text_samples_file_paths: Iterable[str], sender_names: Optional[Iterable[str]] = None
    ) -> None:
        super().__init__(text_samples_file_paths)

        self.sender_names = None if not sender_names else set(sender_names)
        self.message_dumps = []

    def load(self) -> None:
        for message_json_file_path in self.text_samples_file_paths:
            with open(message_json_file_path, "rt") as in_file:
                message_dump = json.load(in_file)
            self.message_dumps.append(message_dump)

    def read_samples_after_load(self) -> Iterator[str]:
        for text_sample_file_path, message_dump in zip(
            self.text_samples_file_paths, self.message_dumps
        ):
            try:
                yield from self._find_messages_from_senders(message_dump)
            except UnicodeError as ex:
                raise InvalidTextSampleFile(
                    f"could not read messages from file '{text_sample_file_path}'"
                ) from ex

    def _find_messages_from_senders(self, messages_dump: dict[str, Any]) -> Iterator[str]:
        for message in messages_dump["messages"]:
            content = message.get("content")
            if content is None:
                continue

            if self.sender_names:
                current_sender_name = self._fix_encoding(message["sender_name"])
                if current_sender_name not in self.sender_names:
                    continue

            message_text = self._fix_encoding(message["content"])
            if message_text:
                yield message_text

    @staticmethod
    def _fix_encoding(text: str) -> str:
        return text.encode("latin1").decode("utf-8")


class SubtitlesReader(TextSampleReader):
    FILE_NAME_PATTERN = "*.xml"

    def __init__(self, text_samples_file_paths: Iterable[str]) -> None:
        super().__init__(text_samples_file_paths)

        self.subtitles = []

    def load(self) -> None:
        for subtitles_file_path in self.text_samples_file_paths:
            try:
                self.subtitles.extend(self._iter_subtitles_from_xml(subtitles_file_path))
            except ParseError as ex:
                raise InvalidTextSampleFile(
                    f"could not load subtitles file '{subtitles_file_path}'"
                ) from ex

    def read_samples_after_load(self) -> Iterator[str]:
        yield from iter(self.subtitles)

    @staticmethod
    def _iter_subtitles_from_xml(subtitles_file_path: str) -> Iterator[str]:
        with open(subtitles_file_path, "rt") as xml_file:
            root = ET.fromstring(xml_file.read())

        for s_tag in root.iter("s"):
            subtitle = "".join(s_tag.itertext()).strip()
            yield subtitle
