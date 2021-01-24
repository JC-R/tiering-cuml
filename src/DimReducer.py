from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem
from time import process_time
import logging
import numpy as np

# using NVIDIA GPU UMAP
from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.metrics import trustworthiness


class DimReducer:

    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=50, metric='cosine', epochs=400, learn_rate=0.05):
        self.umap = cumlUMAP(n_neighbors=n_neighbors,
                             n_components=n_components,
                             n_epochs=epochs,
                             learning_rate=learn_rate,
                             output_type="cudf")
        self.logger = logging.getLogger(__name__)

    # dataset should already be on gpu for efficiency
    def fit_transform(self, data):
        t1 = process_time()
        result = self.umap.fit_transform(data)
        t2 = process_time()
        self.logger.debug("UMAP Fit: %d samples, %d sec", data.shape[0], t2 - t1)
        gpu_mem(0)
        return result

    # create dim reduce model
    def fit(self, data):
        t1 = process_time()
        self.umap.fit(data)
        t2 = process_time()
        self.logger.debug("Fit: %d samples, %d sec", self.umap.X_m.shape[0], t2 - t1)
        gpu_mem(0)
        return self

    def reduce(self, data, as_df=True):
        t1 = process_time()
        result = self.umap.transform(data)
        t2 = process_time()
        self.logger.debug("Reduce: %d samples, %d sec", data.shape[0], t2 - t1)
        gpu_mem(0)
        if as_df:
            return result
        else:
            return result.as_matrix()

    def quality(self, data, data_embedded):
        return trustworthiness(data, data_embedded)

    def save_model(self, fname):
        np.dump(self.umap, fname)
        return self


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    gpu_mem(0)
    reducer = DimReducer()
    train_data, test_data = read("/mnt/m2-1/cw09b.dh.1k.spam.70.dochits.tier.5.t5-base.embeddings.0.npz",
                                 sampling=True,
                                 train_size=1000000,
                                 test_size=None)

    # reducer.fit(train_data)
    # data = reducer.reduce(train_data)
    data = reducer.fit(train_data).transform()
    gpu_mem(0)
