# Raster clustering
## Usage
### Overview
1. Choose your raster files
2. Configure nodata and scaling handling
3. Configure the [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) clustering algorithm
    - Or "number of clusters" or "distance threshold"

```bash
cd fire2a-lib/cluster-tmp
python aggcluster.py -d 10.0

# windows💩
C:\\PROGRA~1\\QGIS33~1.3\\bin\\python-qgis.bat aggcluster.py -d 10.0
```
[how to: windows 💩 use qgis-python](https://github.com/fire2a/fire2a-lib/tree/main/qgis-launchers)

### 1. Choose your raster files
- Any [GDAL compatible](https://gdal.org/en/latest/drivers/raster/index.html) raster will be read
- Place them all in the same directory where the script will be executed
- "Quote them" if they have any non alphanumerical chars [a-zA-Z0-9]

### 2. Preprocessing configuration
1. (Optional) __nodata__ read values are replaced by `np.nan`. An 3x3 image filter (taking its 8-neighbors) is performed only on them. Replacing then with one of the following methods:
- "no": Do not handle missing data (Or skip the line)
- "mean": (Default) Replace missing values with the mean of the non-nan neighborhood.
- "median": Replace missing values with the median of the non-nan neighborhood.
- "mode": Replace missing values with the mode of the non-nan neighborhood.
- "min": Replace missing values with the minimum value of the non-nan neighborhood.
- "max": Replace missing values with the maximum value of the non-nan neighborhood.

2. SCALING (Optional). Specifies the method to scale the data. Possible values are:
- "no": Do not scale the data.
- "onehot": Apply [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to Categorical data.
- "robust": (Default) Apply [Robust Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to regular data with outliers.
- "standard": Apply [Standard Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) (x-mean)/stddev to the data.

A compatible toml text file is needed, specifying for each file:
```toml
["file name.tif"]
nodata = "mean"
scaling = "robust"
```
A sample `config.toml` is provided in the repository.

### 3. Clustering configuration
1. __Agglomerative__ clustering algorithm is used. The following parameters are muttually exclusive:
- n_clusters: The number of clusters to form as well as the number of centroids to generate.
- distance_threshold: The linkage distance threshold above which, clusters will not be merged. When scaling start with 10.0 and downward (0.0 is compute the whole algorithm).

For passing more parameters, see [here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
