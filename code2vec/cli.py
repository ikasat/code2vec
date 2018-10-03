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
@click.option('--mode', '-m', default='method')
def document_split(mode):
    import sys
    import json
    from .splitter import convert_jmethdeps_to_words_map
    from .jsons import parse_json_stream
    for jmethdeps in parse_json_stream(sys.stdin):
        words_map = convert_jmethdeps_to_words_map(jmethdeps, mode=mode)
        json.dump(words_map, sys.stdout)

# word2vec

@main.group(invoke_without_command=True)
@click.pass_context
def word2vec(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

@word2vec.command('train')
@click.option('--model-file', '-f', default='word2vec.model')
def word2vec_train(model_file):
    import gensim
    import sys
    from .jsons import parse_json_stream
    words_list = []
    for words_map in parse_json_stream(sys.stdin):
        words_list += words_map.values()
    print('training...', file=sys.stderr)
    w2v = gensim.models.word2vec.Word2Vec(words_list)
    w2v.train(words_list, total_examples=len(words_list), epochs=30)
    w2v.save(model_file)

@word2vec.command('train-dir')
@click.option('--model-file', '-f', default='word2vec.model')
@click.argument('dir')
def word2vec_train_dir(model_file, dir):
    import gensim
    sentences = gensim.models.word2vec.PathLineSentences(dir)
    w2v = gensim.models.word2vec.Word2Vec(sentences)
    w2v.save(model_file)

@word2vec.command('predict')
@click.option('--model-file', '-f', default='word2vec.model')
@click.option('--history-file', '-H', default='predict-shell.hist')
def word2vec_predict(model_file, history_file):
    import gensim
    from .predict import shell
    w2v = gensim.models.word2vec.Word2Vec.load(model_file)
    shell(w2v, history_file)

@word2vec.command('predict-args')
@click.option('--model-file', '-f', default='word2vec.model')
@click.argument('words', nargs=-1)
def word2vec_predict_args(model_file, words):
    import gensim
    from .predict import predict
    w2v = gensim.models.word2vec.Word2Vec.load(model_file)
    predict(w2v, words)

# fasttext

@main.group(invoke_without_command=True)
@click.pass_context
def fasttext(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

@fasttext.command('train')
@click.option('--model-file', '-f', default='fasttext.model')
def fasttext_train(model_file):
    import gensim
    import sys
    from .jsons import parse_json_stream
    words_list = []
    for words_map in parse_json_stream(sys.stdin):
        words_list += words_map.values()
    print('training...', file=sys.stderr)
    ftxt = gensim.models.FastText(words_list)
    # ftxt.train(words_list, total_examples=len(words_list), epochs=30)
    ftxt.save(model_file)

@fasttext.command('predict')
@click.option('--model-file', '-f', default='fasttext.model')
@click.option('--history-file', '-H', default='predict-shell.hist')
def fasttext_predict(model_file, history_file):
    import gensim
    from .predict import shell
    ftxt = gensim.models.FastText.load(model_file)
    shell(ftxt, history_file)

@fasttext.command('predict-args')
@click.option('--model-file', '-f', default='fasttext.model')
@click.argument('words', nargs=-1)
def fasttext_predict_args(model_file, words):
    import gensim
    from .predict import predict
    ftxt = gensim.models.FastText.load(model_file)
    predict(ftxt, words)

if __name__ == '__main__':
    main()
