import numpy as np
from hashlib import sha256



def create_rng(seed_string: str):
    hash = sha256(seed_string.encode('utf-8'))
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    return np.random.default_rng(seed)
