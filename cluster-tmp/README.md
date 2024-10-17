# Raster clustering
## Usage
### Overview
1. Choose your raster files
2. Configure nodata and scaling strategies in the `config.toml` file
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
See the `config.toml` file for example of the configuration of the preprocessing steps. The file is structured as follows:

```toml
["filename.tif"]
no_data_strategy = "most_frequent"
scaling_strategy = "onehot"
```

1. __scaling_strategy__
   - can be "standard", "robust", "onehot" (default is robust, onehot is categorical)
   - default is "robust"
   - [Standard](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): (x-mean)/stddev
   - [Robust](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): same but droping the tails of the distribution
   - [OneHot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): __for CATEGORICAL DATA__

2. __no_data_strategy__ 
   - can be "mean", "median", "most_frequent", "constant"
   - default is "mean"
   - categorical data should use "most_frequent" or "constant"
   - "constant" will use the value in __fill_value__ (see below)
   - [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

3. __fill_value__ 
   - used when no_data_strategy is "constant"
   - default is 0
   - [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)


### 3. Clustering configuration
1. __Agglomerative__ clustering algorithm is used. The following parameters are muttually exclusive:
- n_clusters: The number of clusters to form as well as the number of centroids to generate.
- distance_threshold: The linkage distance threshold above which, clusters will not be merged. When scaling start with 10.0 and downward (0.0 is compute the whole algorithm).

For passing more parameters, see [here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
