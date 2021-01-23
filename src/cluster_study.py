# Both import methods supported
from src.DimReducer import DimReducer
from src.utils.file_utils import read
from src.utils.gpu_utils import gpu_mem, to_gpu
from cuml import DBSCAN

import logging
import mlflow


# load dataset, scale and cache
def load_dataset_to_gpu(filename, sampling=True, train_size=1000000, test_size=None):
    logging.info("Down-sampling input data to %d", train_size)
    train_data, test_data = read(filename,
                                 sampling=True,
                                 train_size=train_size,
                                 test_size=None)
    gdf = to_gpu(train_data)
    gpu_mem(0)
    return gdf


def dim_reduce(gdf, ndims=5, n_neighbors=15):
    reducer = DimReducer(n_components=ndims, n_neighbors=n_neighbors)
    data = reducer.fit_transform(gdf)
    gpu_mem(0)
    return data


# cluster
def cluster(gdf, eps, minSamples):
    # cpu clustering
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
    # clusterer.fit(data)

    logging.info("cuml.SCANDB Clustering - eps=%0.3f, samples=%d", eps, minSamples)
    # GPU clustering
    result = DBSCAN(eps=eps,
                    min_samples=minSamples,
                    verbose=False,
                    calc_core_sample_indices=True,
                    output_type='cudf').fit(gdf)

    metric1 = result.labels_.max()
    metric2 = len(result.core_sample_indices_)

    mlflow.log_metric('labels', metric1)
    mlflow.log_metric('core_samples', metric2)

    gpu_mem(0)
    logging.debug("cuml.DBSCAN - dims: %d, max distance: %0.3f, min samples: %d, labels: %d, core_samples: %d",
                  ndims, eps, minSamples, metric1, metric2)


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

# run/log experiment
logging.info("MLFlow at %s: %s", mlflow.get_tracking_uri(), 't5 dim/clustering study')

try:
    experiment_id = mlflow.create_experiment("T5 dim/cluster study")
except:
    experiment_id = mlflow.set_experiment("T5 dim/cluster study")

#  load input data, send to GPU
logging.info("Loading input data")
train_size = 1000000
gdf = load_dataset_to_gpu(filename="/mnt/m2-1/cw09b.dh.1k.spam.70.dochits.tier.5.t5-base.embeddings.0.npz",
                          sampling=True,
                          train_size=train_size)

#  reduce dimensions
for ndims in range(5, 50, 5):

    # ndims=5, n_neighbors = 15, (in top2vec)
    logging.info("GPU-UMAP Dimensionality reduction: %d to %d, min neighbors=%d",
                 gdf.shape[1],
                 ndims,
                 15)

    embedding = dim_reduce(gdf, ndims, 15)

    for minSamples in range(ndims, ndims * 10, ndims):
        for e in range(1, 40, 1):
            eps = e * 1.0 / 100
            # eps = 0.05
            # minSamples = 10

            with mlflow.start_run(experiment_id=experiment_id) as run:
                mlflow.log_param('corpus', 'cw09b')
                mlflow.log_param('dh', '1k')
                mlflow.log_param('model', 't5-base')
                mlflow.log_param('spam', 70)
                mlflow.log_param('tier', 40)
                mlflow.log_param('ordering', 'dh')
                mlflow.log_param('train_data', train_size)
                mlflow.log_param('dims', ndims)
                mlflow.log_param('neighbors', 15)
                mlflow.log_param('dims', ndims)
                mlflow.log_param('neighbors', 15)
                mlflow.log_param('eps', eps)
                mlflow.log_param('min_samples', minSamples)

                cluster(embedding, eps, minSamples)

    del embedding
