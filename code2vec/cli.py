import click


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


# document


@main.group(invoke_without_command=True)
@click.pass_context
def document(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


@document.command('split')
@click.option('--mode', '-m', default='class')
def document_split(mode):
    import sys
    import json
    from .splitter import convert_jmethdeps_to_sentences_map
    from .jsons import parse_json_stream
    for jmethdeps in parse_json_stream(sys.stdin):
        sentences_map = convert_jmethdeps_to_sentences_map(jmethdeps, mode=mode)
        json.dump(sentences_map, sys.stdout)


@document.command('split-dir')
@click.argument('dir')
@click.option('--mode', '-m', default='class')
@click.option('--gz/--no-gz', '-g/-G')
def document_split_dir(dir, mode, gz):
    import sys
    import os
    import urllib
    import gzip
    from .splitter import convert_jmethdeps_to_sentences_map
    from .jsons import parse_json_stream
    for jmethdeps in parse_json_stream(sys.stdin):
        sentences_map = convert_jmethdeps_to_sentences_map(jmethdeps, mode=mode)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for key, sentences in sentences_map.items():
            ext = '.gz' if gz else '.txt'
            basename = urllib.parse.quote(key) + ext
            filepath = os.path.join(dir, basename)
            if gz:
                with gzip.open(filepath, 'wb') as f:
                    for sentence in sentences:
                        f.write((' '.join(sentence) + '\n').encode('utf-8'))
            else:
                with open(filepath, 'w') as f:
                    for sentence in sentences:
                        f.write(' '.join(sentence) + '\n')


# embed


def _get_model_class(name):
    import gensim
    model_class_map = {
        'word2vec': gensim.models.word2vec.Word2Vec,
        'fasttext': gensim.models.FastText,
    }
    return model_class_map[name.lower()]


@main.group(invoke_without_command=True)
@click.pass_context
def embed(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


@embed.command('train')
@click.option('--model', '-m', 'model_name', default='word2vec')
@click.option('--model-file', '-f', default='embed.model')
@click.option('--epochs', '-e', default=5)
def embed_train(model_name, model_file, epochs):
    import sys
    from .jsons import parse_json_stream
    model_class = _get_model_class(model_name)
    sentences = []
    for sentences_map in parse_json_stream(sys.stdin):
        for wl in sentences_map.values():
            sentences.extend(wl)
    model = model_class(sentences, iter=epochs)
    model.save(model_file)


@embed.command('train-dir')
@click.argument('dir')
@click.option('--model', '-m', 'model_name', default='word2vec')
@click.option('--model-file', '-f', default='embed.model')
@click.option('--epochs', '-e', default=5)
def embed_train_dir(dir, model_name, model_file, epochs):
    import gensim
    model_class = _get_model_class(model_name)
    sentences = gensim.models.word2vec.PathLineSentences(dir)
    model = model_class(sentences, iter=epochs)
    model.save(model_file)


@embed.command('predict')
@click.option('--model', '-m', 'model_name', default='word2vec')
@click.option('--model-file', '-f', default='embed.model')
@click.option('--history-file', '-H', default='predict-shell.histfile')
def embed_predict(model_name, model_file, history_file):
    from .predict import shell
    model = _get_model_class(model_name).load(model_file)
    shell(model, history_file)


@embed.command('predict-args')
@click.argument('word', nargs=-1)
@click.option('--model', '-m', 'model_name', default='word2vec')
@click.option('--model-file', '-f', default='embed.model')
def word2vec_predict_args(word, model_name, model_file):
    from .predict import predict
    model = _get_model_class(model_name).load(model_file)
    predict(model, word)
