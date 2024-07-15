import string
import re


def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.replace("  ", ' ')
    return s.strip()


def make_prompt_safe(s):
    return s.replace("{", "{{").replace("}", "}}")