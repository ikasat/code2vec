# https://stackoverflow.com/questions/27879958/reading-large-json-file-in-python-raw-decode
import functools
import json


def parse_json_stream(jfile, *, buffersize=4096 * 1024):
    decoder = json.JSONDecoder()
    buffer = ''
    for chunk in iter(functools.partial(jfile.read, buffersize), ''):
        buffer += chunk
        while buffer:
            try:
                result, index = decoder.raw_decode(buffer)
                yield result
                buffer = buffer[index:]
            except ValueError:
                break
