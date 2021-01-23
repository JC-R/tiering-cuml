from cuml.cluster import DBSCAN

from src.DimReducer import DimReducer
from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem

import logging


class T5_DBSCAN:

    def __init__(self, ndims=5, nn=25, eps=0.1, minSamples=25, coreSamples=True, verbose=False):
        self.logger = logging.getLogger("T5Clustering")
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
        self.reducer = DimReducer(n_components=ndims, n_neighbors=nn)
        gpu_mem(0)
        self.train_data = None
        self.test_data = None
        self.data = None
        self.clusterer = DBSCAN(eps=eps, min_samples=minSamples,
                                verbose=verbose,
                                calc_core_sample_indices=coreSamples,
                                output_type='cudf')

    def split(self, fname, sampling=True, train_size=1000000, test_size=None):
        self.train_data, self.test_data = read(fname, sampling=sampling, train_size=train_size, test_size=test_size)
        return self

    def reduce(self, train_data):
        self.logger.info("Dimensionality reduction (UMAP): %s", self.reducer.reducer.get_params)
        self.data = self.reducer.fit_transform(train_data)
        return self

    def cluster(self):
        self.logger.info("Clustering ... (DBSCAN)")
        gpu_mem(0)
        self.clusterer.fit(self.data)
        self.logger.info("Clusters: %d, Core samples: %d", self.clusterer.labels_.max(),
                             len(self.clusterer.core_sample_indices_))
