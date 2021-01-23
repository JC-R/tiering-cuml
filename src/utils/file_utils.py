from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from src.utils.read_embeddings import read_embeddings
import logging


def read(filename, sampling=True, train_size=1000000, test_size=None, scale=True):
    test_data = None
    t1 = process_time()
    if scale:
        logging.info("Scaling the input...")
        dataset = StandardScaler().fit_transform(read_embeddings(filename))
    else:
        dataset = read_embeddings(filename)
    if sampling:
        if test_size is not None:
            logging.info("Sampling training/test sets to %d/%d", train_size, test_size)
            train_data, test_data = train_test_split(dataset, train_size=train_size, test_size=test_size)
        else:
            logging.info("Sampling training set to %d", train_size)
            train_data = shuffle(dataset, n_samples=train_size)
    else:
        train_data = dataset
    del dataset
    t2 = process_time()
    logging.debug("Reading/split took %d sec", t2 - t1)
    return train_data, test_data
