import code2vec.splitter as spl
from collections import Counter

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

words_map_01 = {
    'com.example.HelloWorld.main': 'array list list add all'.split(),
    'com.example.HelloWorld.sub': 'inner f'.split(),
    'com.example.HelloWorld.Inner.f': 'hash map map put'.split(),
}

words_map_class_01 = {
    'com.example.HelloWorld': 'array list list add all inner f'.split(),
    'com.example.HelloWorld.Inner': 'hash map map put'.split(),
}

def test_convert_jmethdeps_to_words_map():
    words_map = spl.convert_jmethdeps_to_words_map(jmethdeps_01)
    for k, words in words_map.items():
        assert Counter(words) == Counter(words_map_01[k])

def test_convert_jmethdeps_to_class_words_map():
    words_map = spl.convert_jmethdeps_to_words_map(jmethdeps_01, mode='class')
    for k, words in words_map.items():
        assert Counter(words) == Counter(words_map_class_01[k])
