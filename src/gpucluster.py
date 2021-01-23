# Both import methods supported
from cuml.cluster import DBSCAN

from src.DimReducer import DimReducer
from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem

import logging

logger = logging.getLogger(__name__)

logger.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
gpu_mem(0)
train_data, test_data = read("/mnt/m2-1/cw09b.dh.1k.spam.70.dochits.tier.5.t5-base.embeddings.0.npz",
                             sampling=True,
                             train_size=1000000,
                             test_size=None)

#  reduce dimensions
ndims = 5
nn = 25
logger.info("Dimensionality reduction: 768 to %d... (UMAP)", nn)
reducer = DimReducer(n_components=ndims, n_neighbors=nn)
data = reducer.fit_transform(train_data)

# cluster
eps = 0.05
minSamples = 25
logger.info("Clustering ... (DBSCAN)")
dbscan = DBSCAN(eps=eps, min_samples=minSamples, verbose=False,
                # max_mbytes_per_batch=4096,
                calc_core_sample_indices=True,
                output_type='cudf')
dbscan.fit(data)
gpu_mem(0)

print(dbscan.labels_.max())
# print(len(dbscan.core_sample_indices_))
# print(dbscan.core_sample_indices_)

