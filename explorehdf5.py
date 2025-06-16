import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1. Open the file (read-only)
with h5py.File("C:\\Users\\Panoptic System\\Downloads\\bert-large-cased_WordNet_Gloss_Corpus.hdf5", 'r') as f: