import pynvml
import cudf
import logging
import pandas as pd


def gpu_mem(device=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)  # Need to specify GPU
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gb = 1024.0 * 1024 * 1024
    logging.debug("GPU memory: free %0.1fGb. used %0.1fGb. total %0.1fGb", mem.free / gb, mem.used / gb, mem.total / gb)
    return mem


# convert numpy array to cuDF dataframe
def np2cudf(df):
    df = pd.DataFrame({'fea%d' % i:df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
      pdf[str(c)] = df[column]
    return pdf


# ingest an ndarray
def to_gpu(data):
    pdf = cudf.DataFrame()
    for i in range(data.shape[1]):
        pdf['fea%d' % i] = data[:, i]
    return pdf
