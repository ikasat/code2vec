import io

import code2vec.jsons


def test_parse_json_stream():
    f = io.StringIO("""{}{"a":"b"}{"c":42}{"d": ["e"]}""" * 7)
    expected = [{}, {'a': 'b'}, {'c': 42}, {'d': ['e']}]
    for i, obj in enumerate(code2vec.jsons.parse_json_stream(f)):
        assert obj == expected[i % len(expected)]
