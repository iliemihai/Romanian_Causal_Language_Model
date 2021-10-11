import re
import string

from regexes.email import EMAIL_REGEX
from regexes.latin import LATIN_REGEX, LATIN_WITH_SPECIAL_REGEX
from regexes.phone import PHONE_REGEX
from regexes.quote import DOUBLE_QUOTE_REGEX, SINGLE_QUOTE_REGEX
from regexes.url import URL_REGEX
from regexes.punk import PUNK_REGEX

allowed_char = string.ascii_letters + string.digits + ':. -'

def make_trans(list_a, list_b):
    return dict((ord(a), b) for a, b in zip(list_a, list_b))

def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

def clean_url(text):
    #remove html tags
    text = re.sub('<.*?>', '', text)

    # removing normal(without space urls)
    text = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', "",text)

    # removing urls that contains space
    result = ''
    for char in text:
        if char in allowed_char:
            result += char
    result = result.split(":")
    for phrase in result:
        p = phrase
        if "//" in p:
            if ('https :' + p) in text:
                text = text.replace('https :' + p, '')
            elif ('http :' + p) in text:
                text = text.replace('http :' + p, '')
        elif '@' in p:
            if p in text:
                text = text.replace(p, '')
    
    return text

def normalize(text, zwnj="\u200c"):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\u200c+", "\u200c", text)
    text = text.replace("▫", "")

    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    text = clean_url(text)
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)
    text = PHONE_REGEX.sub(r" \1 ", text)
    #text = LATIN_REGEX.sub(r" \1 ", text)

    text = text.replace(f" {zwnj} ", f"{zwnj}")
    text = text.replace(f"{zwnj} ", f"{zwnj}")
    text = text.replace(f" {zwnj}", f"{zwnj}")

    return text

    if __name__ == "__main__":

        input_text = "  "
        input_text = normalize(input_text)

        print(textwrap.fill(input_text))
        print(normalize(input_text))
