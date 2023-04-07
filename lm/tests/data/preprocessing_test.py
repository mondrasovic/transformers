import pytest

from lm.data.preprocessing import ConversationPreprocessor


@pytest.mark.unit
class TestConversationPreprocessor:
    @pytest.fixture
    def conversation_preprocessor(self):
        return ConversationPreprocessor()
    
    @pytest.mark.parametrize("text, expected_text", [
        pytest.param(
            "  - neviem,     čo tu napísať, ale snáď to bude FUNGOVAŤ! :-D :-)      ",
            "neviem, co tu napisat, ale snad to bude FUNGOVAT! :D :)",
            id="mixed_text_accents_emojis_whitespace"
        )
    ])
    def test_preprocess(self, conversation_preprocessor, text, expected_text):
        assert conversation_preprocessor.preprocess(text) == expected_text
    
    @pytest.mark.parametrize("text, expected_text", [
        pytest.param(
            "??? :-D :D __ :-) :) || :-( :( :-P :P :-* :*!!!",
            "??? :D :D __ :) :) || :( :( :P :P :* :*!!!",
            id="all_emojis_with_fillers"
        )
    ])
    def test_normalize_emojis(self, text, expected_text):
        assert ConversationPreprocessor.normalize_emojis(text) == expected_text

    @pytest.mark.parametrize("text, expected_text", [
        pytest.param(
            "áÁäéÉčČďĎíÍľĽňŇóÓôřŘšŠťŤúÚýÝžŽ",
            "aAaeEcCdDiIlLnNoOorRsStTuUyYzZ",
            id="all_possible_characters"
        )
    ])
    def test_remove_accents(self, text, expected_text):
        assert ConversationPreprocessor.remove_accents(text) == expected_text


    @pytest.mark.parametrize("text, expected_text", [
        pytest.param(
            "- something",
            "something",
            id="no_leading_one_trailing_space"
        ),
        pytest.param(
            " - something",
            "something",
            id="one_leading_one_trailing_space"
        ),
        pytest.param(
            "-something",
            "something",
            id="no_leading_no_trailing_space"
        ),
        pytest.param(
            "  -   something",
            "something",
            id="multiple_leading_multiple_trailing_spaces"
        ),
    ])
    def test_remove_leading_dash(self, text, expected_text):
        assert ConversationPreprocessor.remove_leading_dash(text) == expected_text


    @pytest.mark.parametrize("text, expected_text", [
        pytest.param(
            " some     words   that contain     multiple\t\twhitespace\tcharacters",
            " some words that contain multiple whitespace characters",
            id="different_no_of_whitespace"
        )
    ])
    def test_normalize_whitespace(self, text, expected_text):
        assert ConversationPreprocessor.normalize_whitespaces(text) == expected_text
