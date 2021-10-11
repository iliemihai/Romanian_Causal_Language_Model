import re

from normalizer import normalize

def normalizer(text, do_lowercae=False):
    text = normalize(text)

    if do_lowercae:
        text = text.lower()

    return text