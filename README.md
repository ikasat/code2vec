# code2vec

## Installation

```sh
git clone --depth 1 https://github.com/ikasat/code2vec.git
cd code2vec/
python3 -m venv .venv
source ./.venv/bin/activate
pip install .
```

## Usage

### Training

See also: https://github.com/ikasat/jmethdeps

```sh
cd <working directory>/
find /path/to/libs -name '*.jar' | xargs -r jmethdeps -j -d | gzip -c >dependency.jsons.gz
zcat dependency.jsons.gz | code2vec document split | gzip -c >documents.jsons.gz
zcat documents.jsons.gz | code2vec embed train -m word2vec -f embed.model -e 30
```

### Prediction

```
$ code2vec embed predict -m word2vec -f embed.model
> add - list + map
0.5713 put
0.5061 contains
0.4275 hash
0.4077 remove
0.3758 bi
0.3743 dictionary
0.3733 absent
0.3674 concurrent
0.3600 filtered
0.3562 entry
>
```
