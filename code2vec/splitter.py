import re
from typing import Dict, List, Tuple


def _convert_camel_to_snake(s: str) -> str:
    t = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', t).lower()


_SPLIT_NAME_RE = re.compile(r'[_\.]')


def split_name(s: str) -> List[str]:
    return [w for w in _SPLIT_NAME_RE.split(_convert_camel_to_snake(s)) if w.isalpha()]


def _parse_fqn(fqn: str) -> Tuple[str, str, str]:
    i = fqn.rfind('.')
    class_name = fqn[:i]
    method_name = fqn[i + 1:]
    j = max(class_name.rfind('.'), class_name.rfind('$'))
    simple_class_name = class_name[j + 1:]
    return (class_name, simple_class_name, method_name)


def convert_dests_to_sentence(dests: List[str]) -> List[str]:
    sentence = []  # type: List[str]
    for dest in dests:
        _c, simple_class_name, method_name = _parse_fqn(dest)
        if not simple_class_name.isnumeric():
            sentence += split_name(simple_class_name)
        sentence += split_name(method_name)
    return sentence


def convert_jmethdeps_to_sentences_map(jmethdeps: Dict[str, List[str]],
                                       mode='method',
                                       enable_sort=True,
                                       allow_empty_sentence=False) -> Dict[str, List[List[str]]]:
    sentences_map = {}  # type: Dict[str, List[List[str]]]
    if enable_sort:
        items = sorted(jmethdeps.items())
    else:
        items = jmethdeps.items()
    for src, dests in items:
        if mode == 'class':
            class_name, _sc, _m = _parse_fqn(src)
            sentences = sentences_map.get(class_name)
            if sentences is None:
                sentences = []
                sentences_map[class_name] = sentences
            sentence = convert_dests_to_sentence(dests)
            if sentence or allow_empty_sentence:
                sentences.append(sentence)
        else:
            sentences_map[src] = [convert_dests_to_sentence(dests)]
    return sentences_map
