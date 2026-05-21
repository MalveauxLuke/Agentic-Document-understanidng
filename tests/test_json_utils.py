from sleuth.utils.json_utils import extract_json_from_text


def test_parse_direct_json():
    assert extract_json_from_text('{"answer": "yes"}') == {"answer": "yes"}


def test_parse_fenced_json():
    text = '```json\n{"answer": "yes"}\n```'
    assert extract_json_from_text(text) == {"answer": "yes"}


def test_parse_json_with_surrounding_text():
    text = 'Here is the result: {"answer": "yes", "nested": {"x": 1}} thanks.'
    assert extract_json_from_text(text) == {"answer": "yes", "nested": {"x": 1}}
