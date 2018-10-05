from collections import Counter

import code2vec.splitter as spl


def _split_to_sentences(s):
    return [l.split() for l in s.split('\n')]


jmethdeps_01 = {
    'com.example.HelloWorld.main': [
        'java.util.ArrayList.<init>',
        'java.util.List.addAll',
    ],
    'com.example.HelloWorld.sub': [
        'com.example.HelloWorld.Inner.f',
    ],
    'com.example.HelloWorld.Inner.f': [
        'java.util.HashMap.<init>',
        'java.util.Map.put',
    ],
}

sentences_map_01 = {
    'com.example.HelloWorld.main': _split_to_sentences('array list list add all'),
    'com.example.HelloWorld.sub': _split_to_sentences('inner f'),
    'com.example.HelloWorld.Inner.f': _split_to_sentences('hash map map put'),
}

sentences_map_class_01 = {
    'com.example.HelloWorld': _split_to_sentences('''array list list add all
inner f'''),
    'com.example.HelloWorld.Inner': _split_to_sentences('hash map map put'),
}


def test_convert_jmethdeps_to_sentences_map():
    sentences_map = spl.convert_jmethdeps_to_sentences_map(jmethdeps_01)
    for k, sentences in sentences_map.items():
        for i, sentence in enumerate(sentences):
            assert sentence == sentences_map_01[k][i]


def test_convert_jmethdeps_to_class_sentences_map():
    sentences_map = spl.convert_jmethdeps_to_sentences_map(jmethdeps_01, mode='class')
    for k, sentences in sentences_map.items():
        for i, sentence in enumerate(sentences):
            assert sentence == sentences_map_class_01[k][i]
