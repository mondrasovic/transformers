from __future__ import annotations

from lm.data.preprocessing import TextSamplePreprocessor
from lm.data.reader import TextSampleReader


class PlainTextDatasetGenerator:
    def __init__(self, preprocessor: TextSamplePreprocessor) -> None:
        self.preprocessor = preprocessor

    def generate_from_readers(
        self, output_file_path: str, *text_sample_readers: tuple[TextSampleReader, ...]
    ) -> None:
        with open(output_file_path, "wt") as out_file:
            for text_sample_reader in text_sample_readers:
                for text_sample in text_sample_reader.read_samples():
                    text_sample_preprocessed = self.preprocessor(text_sample)
                    if text_sample_preprocessed:
                        out_file.write(f"{text_sample_preprocessed}\n")
                out_file.flush()
