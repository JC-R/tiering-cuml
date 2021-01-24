from cuml.cluster import KMeans

from src.DimReducer import DimReducer
from src.utils.file_utils import read, split
from src.utils.gpu_utils import to_gpu
import numpy as np
import logging

# create batches from a list/sequence
def group_list(l, group_size):
    """
    :param l: list or sequence
    :param group_size:
    :return: batch
    """
    for i in np.xrange(0, len(l), group_size):
        yield l[i:i + group_size]


class T5_KMEANS:

    def __init__(self, ndims=5, nn=25, k=10000, verbose=False):
        self.logger = logging.getLogger("T5Clustering")
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
        self.ndims = ndims
        self.nn = nn
        self.k = k
        self.data = None
        self.train_data = None
        self.test_data = None
        self.data_embedded = None
        self.sampling = None
        self.reducer = DimReducer(n_components=ndims, n_neighbors=nn)
        self.kmeans = KMeans(n_clusters=k, verbose=verbose)

    def split(self, fname, sampling=False, train_size=1000000, test_size=None):
        self.data = read(fname, scale=True)
        self.sampling = sampling
        if sampling:
            self.train_data, test_data = split(train_size=train_size, test_size=test_size)
        else:
            self.train_data = self.data
        return self

    def reduce(self):
        self.logger.info("Dimensionality reduction (UMAP): %s", self.reducer.umap.get_params)
        self.reducer.fit(self.train_data)
        del self.train_data
        step = min(500000, len(self.data))
        result = self.reducer.reduce(self.data[0:step], as_df=False)
        for i in range(step, len(self.data), step):
            result = np.append(result, self.reducer.reduce(self.data[i:i+step], as_df=False), axis=0)
            self.logger.debug("Dim reduce batch : %d", i)
        self.data = None
        self.data_embedded = result
        return self

    def cluster(self):
        self.logger.info("Clustering ... (KMeans)")
        self.kmeans.fit(self.data_embedded)
        self.logger.info("Clusters: %d", len(self.kmeans.cluster_centers_))
        return self
