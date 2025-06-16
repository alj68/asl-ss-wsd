import h5py
import numpy as np
import matplotlib.pyplot as plt

# 1. Open the file (read-only)
with h5py.File("C:\\Users\\Panoptic System\\Downloads\\bert-large-cased_WordNet_Gloss_Corpus.hdf5", 'r') as f:
    # 2. List top-level groups/datasets
    print("Keys: %s" % list(f.keys()))

    # Suppose you see a dataset called "embeddings":
    dset = f['embeddings']  

    # 3. Check its shape and dtype
    print("Shape:", dset.shape)
    print("Dtype:", dset.dtype)

    # 4. Read it into memory (as a NumPy array)
    data = dset[...]      # data.shape might be (N, D)

# 5. Visualize
#   • If it’s a 1D or 2D array, you can:
if data.ndim == 1:
    plt.plot(data)
    plt.title("1D Series from embeddings")
    plt.xlabel("Index")
    plt.ylabel("Value")
elif data.ndim == 2:
    plt.imshow(data, aspect='auto')
    plt.colorbar(label='Value')
    plt.title("2D Array Heatmap")
plt.show()
