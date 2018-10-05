import atexit
import readline
import traceback


def _show_result(result):
    for word, distance in result:
        print('{:6.4f} {}'.format(distance, word))


def predict(model, words):
    positive_words = []
    negative_words = []
    for word in words:
        if word.startswith('+'):
            positive_words.append(word[1:])
        elif word.startswith('-'):
            negative_words.append(word[1:])
        else:
            positive_words.append(word)
    result = model.most_similar(positive=positive_words, negative=negative_words)
    _show_result(result)


def _parse_line_in_shell(line):
    line = line.strip()
    words_by_pm = {'+': [], '-': []}
    mode = '+'
    pos = 0
    while pos < len(line):
        p = line.find('+', pos)
        p = p if p >= 0 else len(line)
        m = line.find('-', pos)
        m = m if m >= 0 else len(line)
        if p > m:
            words_by_pm[mode] += [w for w in line[pos:m].split() if w]
            mode = '-'
            pos = m + 1
        else:
            words_by_pm[mode] += [w for w in line[pos:p].split() if w]
            mode = '+'
            pos = p + 1
    return words_by_pm['+'], words_by_pm['-']


def shell(model, history_file: str):
    try:
        readline.read_history_file(history_file)
        h_len = readline.get_current_history_length()
    except FileNotFoundError:
        open(history_file, 'wb').close()
        h_len = 0

    def save(prev_h_len, histfile):
        new_h_len = readline.get_current_history_length()
        readline.set_history_length(1000)
        readline.append_history_file(new_h_len - prev_h_len, histfile)

    atexit.register(save, h_len, history_file)
    while True:
        try:
            line = input('> ')
            positive, negative = _parse_line_in_shell(line)
            if len(positive) + len(negative) > 0:
                result = model.most_similar(positive=positive, negative=negative)
                _show_result(result)
        except KeyError as e:
            print(' '.join(e.args))
        except Exception as e:
            traceback.format_exc()
            print()
            break
