from itertools import islice
from typing import TypeVar, Iterator, List, Iterable
from tqdm import tqdm

T = TypeVar('T', covariant=True)


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def get_num_lines(file: str):
    with open(file, "r", encoding="utf-8", errors='ignore') as f:
        return (sum(bl.count("\n") for bl in blocks(f)))


def read_sentences(path: str, progbar=False, encoding='utf-8', strip=True, max_sentences=None, **kwargs) -> Iterator[str]:
    numlines = get_num_lines(path)
    if progbar: print(f'reading {max_sentences if max_sentences else numlines} lines from {path}')

    with open(path, 'r', encoding=encoding, errors='replace') as f:
        # if progbar:
        _idx = 0
        for l in (tqdm(f, total=(numlines if not max_sentences else max_sentences)) if progbar else f):
            if max_sentences and _idx == max_sentences:
                return
            _idx += 1
            yield l.strip() if strip else l.rstrip()


import json


def read_jsonl(path):
    ret = []
    for line in read_sentences(path, progbar=False):
        d = json.loads(line)
        ret.append(d)

    return ret


def write_to_file(s, file, how='a', also_print=False):
    with open(file, how) as f:
        f.write(s + "\n")
    if also_print:
        print(s + '\n')


def batched(xs: Iterator[T], batch_size: int, total: int = 0) -> Iterator[List[T]]:
    buf: List[T] = []

    if total:
        count = 0
        for x in tqdm(xs, total=total):
            if count == total: return
            count += 1
            buf.append(x)
            if len(buf) == batch_size:
                yield buf
                buf.clear()
    else:
        for x in xs:
            buf.append(x)
            if len(buf) == batch_size:
                yield buf
                buf.clear()

    if len(buf) != 0:
        yield buf
