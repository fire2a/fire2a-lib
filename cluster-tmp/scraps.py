def hdbscan_data_list(*args, **kwargs):
    """Perform HDBSCAN clustering on a list of spatial data arrays."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Combine the layers into a single feature space
    data = np.hstack(args)

    # Normalize the combined data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Optionally, reduce dimensionality using PCA
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(normalized_data)

    height, width = args[0].shape

    N = height * width
    if "min_cluster_size" not in kwargs:
        min_cluster_size = int(N * 0.1)
    if "max_cluster_size" not in kwargs:
        max_cluster_size = int(N * 0.5)

    # Perform HDDBSCAN clustering
    clustering = HDBSCAN(n_jobs=30, **kwargs)
    # clustering.fit(reduced_data)
    clustering.fit(normalized_data)
    labels = clustering.labels_

    return labels, clustering


def plot_labels(elevation, trees, age, labels_reshaped):
    import matplotlib.pyplot as plt

    # Create a 4x4 plot to visualize the three layers and the clustering results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot elevation
    axs[0, 0].imshow(elevation, cmap="terrain", interpolation="nearest")
    axs[0, 0].set_title("Elevation")
    axs[0, 0].set_xticks(np.arange(0, width, 5))
    axs[0, 0].set_yticks(np.arange(0, height, 5))
    axs[0, 0].grid(True)

    # Plot tree height
    axs[0, 1].imshow(trees, cmap="YlGn", interpolation="nearest")
    axs[0, 1].set_title("Tree Height")
    axs[0, 1].set_xticks(np.arange(0, width, 5))
    axs[0, 1].set_yticks(np.arange(0, height, 5))
    axs[0, 1].grid(True)

    # Plot age
    axs[1, 0].imshow(age, cmap="cool", interpolation="nearest")
    axs[1, 0].set_title("Age")
    axs[1, 0].set_xticks(np.arange(0, width, 5))
    axs[1, 0].set_yticks(np.arange(0, height, 5))
    axs[1, 0].grid(True)

    # Plot clustering results
    im = axs[1, 1].imshow(labels_reshaped, cmap="viridis", interpolation="nearest")
    axs[1, 1].set_title("Clustering Results")
    axs[1, 1].set_xticks(np.arange(0, width, 5))
    axs[1, 1].set_yticks(np.arange(0, height, 5))
    axs[1, 1].grid(True)

    # Add colorbars
    fig.colorbar(axs[0, 0].imshow(elevation, cmap="terrain", interpolation="nearest"), ax=axs[0, 0])
    fig.colorbar(axs[0, 1].imshow(trees, cmap="YlGn", interpolation="nearest"), ax=axs[0, 1])
    fig.colorbar(axs[1, 0].imshow(age, cmap="cool", interpolation="nearest"), ax=axs[1, 0])
    fig.colorbar(im, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


def make_test_data():
    # Define map dimensions
    width = 160
    height = 90

    # Create random data for the 3 layers
    np.random.seed(0)  # For reproducibility

    # Create elevation data using a 2D normal distribution to emulate slopes
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    elevation = np.exp(-(x**2 + y**2)) * 100  # Elevation with a peak in the center

    # Create tree height and age data using wavy patterns with dampening
    trees = np.exp(-x) * np.cos(y) * 50  # Tree height with a wavy pattern
    shrubs = np.exp(-y * x) * np.sin(x) * 50  # Age with a wavy pattern
    age = np.exp(-y) * np.cos(x) * 50  # Age with a wavy pattern

    # nodatas
    if True:
        elevation[3, 3] = np.nan
        trees[1, 1] = np.nan
        shrubs[2, 1] = np.nan
        age[2, 2] = np.nan
        elevation[:, 0] = np.nan
        trees[0, :] = np.nan
        shrubs[-1, :] = np.nan
        age[:, 0] = np.nan

        elevation = neighbor_nan_filter(elevation, method="mean")
        trees = neighbor_nan_filter(trees, method="median")
        shrubs = neighbor_nan_filter(shrubs, method="min")
        age = neighbor_nan_filter(age, method="mode")

    if False:
        minmaxfunc = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        for arr in [elevation, trees, shrubs, age]:
            arr = minmaxfunc(arr)
    else:
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        for arr in [elevation, trees, shrubs, age]:
            arr = scaler.fit_transform(arr)

        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder()
        for arr in [elevation, trees, shrubs, age]:
            arr = enc.fit_transform(arr)

    data_list = [elevation, trees, shrubs, age]
    height, width = check_shapes(data_list)
    labels_reshaped, clustering = agglomerative_clustering_data_list(*data_list, n_clusters=13)
    labels_reshaped, clustering = hdbscan_data_list(*data_list)

    plot_labels(elevation, trees, age, labels_reshaped)


def scaling(arr, method="robust"):
    """
    2. SCALING (Optional). Specifies the method to scale the data. Possible values are:
    - "no": Do not scale the data.
    - "onehot": Apply [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to Categorical data.
    - "robust": (Default) Apply [Robust Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to regular data with outliers.
    - "standard": Apply [Standard Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) (x-mean)/stddev to the data.
    """
    if not method or method == "no":
        return arr
    elif method == "onehot":
        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder()
        return enc.fit_transform(arr)
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        return scaler.fit_transform(arr)
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return scaler.fit_transform(arr)


def agglomerative_clustering_data_list(*args, **kwargs):
    """Perform Agglomerative Clustering with connectivity on a list of spatial data arrays."""
    from sklearn.cluster import AgglomerativeClustering

    # Stack the layers to create a 3D array
    data = np.stack(args, axis=-1)

    # Reshape the data to a 2D array where each row is a point and each column is a feature
    data_reshaped = data.reshape(-1, len(args))

    height, width = args[0].shape
    # Generate the connectivity matrix using the custom callable
    connectivity = adjacency_connectivity(height, width)

    # Perform Agglomerative Clustering with connectivity
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, **kwargs)
    clustering.fit(data_reshaped)
    labels = clustering.labels_

    # Reshape the labels back to the original spatial map shape
    labels_reshaped = labels.reshape(height, width)
    return labels_reshaped, clustering
