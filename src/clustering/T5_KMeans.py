from cuml.cluster import KMeans

from src.DimReducer import DimReducer
from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem
import logging


class T5_KMEANS:

    def __init__(self, ndims=5, nn=25, k=10000, verbose=False):
        self.logger = logging.getLogger("T5Clustering")
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
        self.reducer = DimReducer(n_components=ndims, n_neighbors=nn)
        gpu_mem(0)
        self.train_data = None
        self.test_data = None
        self.data = None
        self.clusterer = KMeans(n_clusters=k, verbose=verbose)

    def split(self, fname, sampling=True, train_size=1000000, test_size=None):
        self.train_data, self.test_data = read(fname, sampling=sampling, train_size=train_size, test_size=test_size)
        return self

    def reduce(self, train_data):
        self.logger.info("Dimensionality reduction (UMAP): %s", self.reducer.reducer.get_params)
        self.data = self.reducer.fit_transform(train_data)
        return self

    def cluster(self):
        self.logger.info("Clustering ... (KMeans)")
        gpu_mem(0)
        self.clusterer.fit(self.data)
        self.logger.info("Clusters: %d", len(self.clusterer.cluster_centers_))
        return self
