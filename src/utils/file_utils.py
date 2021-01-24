from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from src.utils.read_embeddings import read_embeddings
import logging


def read(filename, scale=True):
    logging.info("Reading %s", filename)
    t1 = process_time()
    if scale:
        logging.info("Scaling the input...")
        dataset = StandardScaler().fit_transform(read_embeddings(filename))
    else:
        dataset = read_embeddings(filename)
    t2 = process_time()
    logging.debug("Reading took %d sec", t2 - t1)
    return dataset


def split(dataset, train_size=1000000, test_size=None):
    logging.info("Sampling the input")
    test_data = None
    t1 = process_time()
    if test_size is not None:
        train_data, test_data = train_test_split(dataset, train_size=train_size, test_size=test_size)
    else:
        train_data = shuffle(dataset, n_samples=train_size)
    t2 = process_time()
    logging.debug("Sampling took %d sec", t2 - t1)
    return train_data, test_data
