import click

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

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

@document.command('corpus')
@click.option('--names-file', '-n', default='names.txt')
@click.option('--dictionary-file', '-d', default='dict.txt')
@click.option('--corpus-file', '-c', default='corpus.mm')
@click.option('--corpus-tfidf-file', '-t', default='corpus_tfidf.mm')
@click.option('--corpus-sms-file', '-s', default='corpus_tfidf_sms.mm.index')
def document_corpus(names_file, dictionary_file, corpus_file, corpus_tfidf_file,
                    corpus_sms_file):
    import gensim
    import sys
    from .jsons import parse_json_stream
    names = []
    words_list = []
    for words_map in parse_json_stream(sys.stdin):
        names += words_map.keys()
        words_list += words_map.values()
    print('building...', file=sys.stderr)
    with open(names_file, 'w') as f:
        for i, name in enumerate(names):
            print("{} {}".format(i, name), file=f)
    dictionary = gensim.corpora.Dictionary(words_list)
    dictionary.filter_extremes(no_below=2, no_above=0.1)
    dictionary.save_as_text(dictionary_file)
    corpus = [dictionary.doc2bow(words) for words in words_list]
    gensim.corpora.MmCorpus.serialize(corpus_file, corpus)
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    gensim.corpora.MmCorpus.serialize(corpus_tfidf_file, corpus_tfidf)
    corpus_tfidf = gensim.corpora.MmCorpus(corpus_tfidf_file)  # reload
    sms = gensim.similarities.SparseMatrixSimilarity(corpus_tfidf)
    sms.save(corpus_sms_file)

@document.command('similar')
@click.option('--names-file', '-n', default='names.txt')
@click.option('--dictionary-file', '-d', default='dict.txt')
@click.option('--corpus-sms-file', '-s', default='corpus_tfidf_sms.mm.index')
def document_similar(names_file, dictionary_file, corpus_sms_file):
    import gensim
    import sys
    names_map = {}
    with open(names_file) as f:
        for line in f:
            si, name = line.split()
            names_map[int(si)] = name
    dictionary = gensim.corpora.Dictionary.load_from_text(dictionary_file)
    sms = gensim.similarities.SparseMatrixSimilarity.load(corpus_sms_file)
    bow = dictionary.doc2bow(sys.stdin.read().split())
    for i, score in sorted(enumerate(sms[bow]), key=lambda t: -t[1])[:10]:
        print("{} {}".format(names_map[i], score))

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

@word2vec.command('predict')
@click.option('--model-file', '-f', default='word2vec.model')
@click.argument('words', nargs=-1)
def word2vec_predict(model_file, words):
    import gensim
    import sys
    from .jsons import parse_json_stream
    positive_words = []
    negative_words = []
    for word in words:
        if word.startswith('+'):
            positive_words.append(word[1:])
        elif word.startswith('-'):
            negative_words.append(word[1:])
        else:
            positive_words.append(word)
    w2v = gensim.models.word2vec.Word2Vec.load(model_file)
    result = w2v.most_similar(positive=positive_words, negative=negative_words)
    for word, distance in result:
        print("{:6.4f} {}".format(distance, word))

@main.group(invoke_without_command=True)
@click.pass_context
def doc2vec(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

@doc2vec.command('train')
@click.option('--model-file', '-f', default='doc2vec.model')
def doc2vec_train(model_file):
    import gensim
    import sys
    import pickle
    from .jsons import parse_json_stream
    from .splitter import split_name
    docs = []
    index_to_name_map = {}
    name_to_index_map = {}
    for words_map in parse_json_stream(sys.stdin):
        for name, words in words_map.items():
            j = max(name.rfind('.'), name.rfind('$'))
            simple_class_name = name[j+1:]
            doc = gensim.models.doc2vec.TaggedDocument(words=words, tags=split_name(simple_class_name))
            i = len(docs)
            index_to_name_map[i] = name
            name_to_index_map[name] = i
            docs.append(doc)
    with open('index_name_assoc.pickle', 'wb') as f:
        pickle.dump((index_to_name_map, name_to_index_map), f)
    print('training...', file=sys.stderr)
    d2v = gensim.models.doc2vec.Doc2Vec(docs)
    d2v.train(docs, total_examples=len(docs), epochs=30)
    d2v.save(model_file)

@doc2vec.command('predict')
@click.option('--model-file', '-f', default='doc2vec.model')
@click.argument('hint')
def doc2vec_predict(model_file, hint):
    import gensim
    import sys
    import pickle
    d2v = gensim.models.doc2vec.Doc2Vec.load(model_file)
    with open('index_name_assoc.pickle', 'rb') as f:
        index_to_name_map, name_to_index_map = pickle.load(f)
    try:
        index = int(hint)
    except ValueError:
        index = name_to_index_map[hint]
    print(index_to_name_map[index])
    for tag, score in d2v.docvecs.most_similar(index):
        print("{:6.4f} {}".format(score, tag))

@main.group(invoke_without_command=True)
@click.pass_context
def lda(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())

@lda.command('train')
@click.option('--model-file', '-f', default='lda.model')
@click.option('--dictionary-file', '-d', default='dict.txt')
@click.option('--corpus-file', '-c', default='corpus.mm')
@click.option('--num-topics', '-N', type=int, default=20)
def lda_train(model_file, dictionary_file, corpus_file, num_topics):
    import gensim
    dictionary = gensim.corpora.Dictionary.load_from_text(dictionary_file)
    corpus = gensim.corpora.MmCorpus(corpus_file)
    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus, num_topics=num_topics, id2word=dictionary)
    lda.save(model_file)
    for i in range(num_topics):
        print('{}: {}'.format(i, lda.print_topic(i)))

@lda.command('predict')
@click.option('--model-file', '-f', default='lda.model')
@click.option('--dictionary-file', '-d', default='dict.txt')
@click.argument('words', nargs=-1)
def lda_predict_all(model_file, dictionary_file, words):
    import gensim
    dictionary = gensim.corpora.Dictionary.load_from_text(dictionary_file)
    lda = gensim.models.ldamodel.LdaModel.load(model_file)
    bow = dictionary.doc2bow(words)
    for i, score in sorted(lda[bow], key=lambda t: -t[1]):
        print("{} {} {}".format(i, score, lda.print_topic(i)))

@lda.command('predict-all')
@click.option('--model-file', '-f', default='lda.model')
@click.option('--dictionary-file', '-d', default='dict.txt')
def lda_predict_all(model_file, dictionary_file):
    import gensim
    import sys
    from .jsons import parse_json_stream
    dictionary = gensim.corpora.Dictionary.load_from_text(dictionary_file)
    lda = gensim.models.ldamodel.LdaModel.load(model_file)
    for words_map in parse_json_stream(sys.stdin):
        for name, words in words_map.items():
            bow = dictionary.doc2bow(words)
            for i, score in sorted(lda[bow], key=lambda t: -t[1]):
                print("{} {} {}".format(i, score, name))
                break

if __name__ == '__main__':
    main()
