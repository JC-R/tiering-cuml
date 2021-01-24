import zipfile
import numpy as np
import logging
from sys import getsizeof


def read_embeddings(filename):
    data = []
    idx = 0
    logging.debug("Reading embeddings from file %s", filename)
    with zipfile.ZipFile(filename, mode='r') as z:
        for name in z.namelist():
            with z.open(name, mode='r') as f:
                buff = f.read()
                vec = np.frombuffer(buff, dtype=np.float32)
                assert len(vec)==768
                data.append(vec)
                idx += 1
    logging.info("Embeddings: %d elements, %0.2fMb", len(data), getsizeof(data)/(1024.0*1024))
    return np.array(data)
