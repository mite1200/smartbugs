from sb.llm.models import PreprocessingConfig
from sb.llm.preprocessing import add_line_numbers, preprocess, remove_solidity_comments


def test_comment_removal_preserves_lines_and_strings():
    source = (
        'string a = "https://example.test/*still-string*/"; // remove me\n'
        "uint x = 1; /* block\ncomment */ uint y = 2;\n"
        'string b = "escaped \\" // still-string";\n'
    )
    result = remove_solidity_comments(source)
    assert result.count("\n") == source.count("\n")
    assert len(result) == len(source)
    assert "remove me" not in result
    assert "block" not in result
    assert "comment" not in result
    assert '"https://example.test/*still-string*/"' in result
    assert '"escaped \\" // still-string"' in result


def test_line_numbering_is_correct_and_preserves_line_count():
    result = add_line_numbers("first\n\nthird")
    assert result == "001 | first\n002 | \n003 | third"
    assert len(result.splitlines()) == 3


def test_preprocessing_can_be_disabled():
    source = "contract A {} // retained\n"
    assert preprocess(source, PreprocessingConfig()) == source


def test_combined_preprocessing_uses_comment_free_source():
    source = "contract A { // comment\n}\n"
    result = preprocess(source, PreprocessingConfig(remove_comments=True, add_line_numbers=True))
    assert result.startswith("001 | contract A {           \n002 | }")
