#! /usr/bin/python3

import sys
import argparse

from src.clustering.T5DBScan import T5_DBSCAN
from src.clustering.T5_KMeans import T5_KMEANS

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="[dbscan, kmeans, hdbscan")
    parser.add_argument("--dims", type=int, default=5)
    parser.add_argument("--nn", type=int, default=15)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--minSamples", type=int, default=15)
    parser.add_argument("--k", type=int, default=10000)
    parser.add_argument("--input", type=str)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_name", type=str, default="kmeans_model.pkl")

    args = parser.parse_args()

    if args.model == "kmeans":
        pipeline = T5_KMEANS(ndims=args.dims, nn=args.nn, k=args.k)
    elif args.model == "dbscan":
        pipeline = T5_DBSCAN(ndims=args.dims, nn=args.nn, eps=args.eps, minSamples=args.minSamples)
    elif args.model == "hdbscan":
        sys.exit(-2)
    else:
        sys.exit(-1)

    pipeline.split(args.input, sampling=args.sampling, train_size=args.train_size, test_size=args.test_size)\
        .reduce()\
        .cluster()

    if args.save_model:
        import pickle
        pickle.dump(pipeline.reducer, open(args.model_name + ".umap.model.pkl", 'wb'))
        pickle.dump(pipeline.kmeans, open(args.model_name + '.kmeans.model.pkl', "wb"))
    pipeline.logger.info("Finished")

