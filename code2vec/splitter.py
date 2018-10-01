import re
from typing import Dict, List

def _convert_camel_to_snake(s):
    t = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', t).lower()

_SPLIT_NAME_RE = re.compile(r'[_\.]')

def split_name(s: str) -> List[str]:
    return [w
            for w in _SPLIT_NAME_RE.split(_convert_camel_to_snake(s))
            if w.isalpha()]

def _parse_fqn(fqn):
    i = fqn.rfind('.')
    class_name = fqn[:i]
    method_name = fqn[i+1:]
    j = max(class_name.rfind('.'), class_name.rfind('$'))
    simple_class_name = class_name[j+1:]
    return (class_name, simple_class_name, method_name)


def convert_dests_to_words(dests: List[str]) -> List[str]:
    words = []
    for dest in dests:
        _c, simple_class_name, method_name = _parse_fqn(dest)
        if not simple_class_name.isnumeric():
            words += split_name(simple_class_name)
        words += split_name(method_name)
    return words

def convert_jmethdeps_to_words_map(jmethdeps: Dict[str, List[str]], mode='method') -> Dict[str, List[str]]:
    words_map = {}
    for src, dests in jmethdeps.items():
        if mode == 'class':
            class_name, _sc, _m = _parse_fqn(src)
            words = words_map.get(class_name, [])
            words_map[class_name] = words + convert_dests_to_words(dests)
        else:
            words_map[src] = convert_dests_to_words(dests)
    return words_map
