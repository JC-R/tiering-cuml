from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem, to_gpu
from time import process_time
import logging

# using NVIDIA GPU UMAP
from cuml.manifold.umap import UMAP as cumlUMAP


class DimReducer:

    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=50, metric='cosine', epochs=400, learn_rate=0.05):
        # self.reducer = umap.UMAP(n_neighbors=n_neighbors,
        #                          min_dist=min_dist,
        #                          n_components=n_components,
        #                          metric=metric,
        #                          random_state=42,
        #                          low_memory=False)
        self.reducer = cumlUMAP(n_neighbors=n_neighbors,
                                n_components=n_components,
                                n_epochs=epochs,
                                learning_rate=learn_rate)
        self.logger = logging.getLogger(__name__)

    # dataset MUST be on gpu
    def fit_transform(self, gdf):
        t1 = process_time()
        data = self.reducer.fit_transform(gdf)
        t2 = process_time()
        self.logger.debug("Fit: %d samples, %d sec", data.shape[0], t2 - t1)
        gpu_mem(0)
        return data

    def fit(self, dataset):
        t1 = process_time()
        gdf = to_gpu(dataset)
        self.reducer.fit(gdf)
        t2 = process_time()
        self.logger.debug("Fit: %d samples, %d sec", gdf.shape[0], t2 - t1)
        gpu_mem(0)
        return self

    def reduce(self, data, as_df=True):
        t1 = process_time()
        dataset = self.reducer.transform(to_gpu(data))
        t2 = process_time()
        self.logger.debug("Reduce: %d samples, %d sec", data.shape[0], t2 - t1)
        del gdf
        gpu_mem(0)
        if as_df:
            return dataset
        else:
            return dataset.as_matrix()


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
    data = reducer.fit_transform(train_data)
    gpu_mem(0)
