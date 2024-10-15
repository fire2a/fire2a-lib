import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Sample data
data_list = [
    np.array([[1, 2, 3, 3], [2, 4, 5, 5], [3, 5, 6, 6]]),  # robust
    np.array([[10, 20, 30, 30], [20, 22, 45, 45], [30, 55, 66, 45]]),  # standard
    np.array([["A", "B", "C", "C"], ["D", "E", "F", "F"], ["A", "E", "E", "C"]]),  # categorical
    np.array([["a", "b", "c", "c"], ["d", "e", "f", "f"], ["a", "e", "e", "d"]]),  # categorical
    np.array([[101, 220, 330, 330], [220, 212, 435, 435], [430, 555, 616, 616]]),  # standard
    np.array([[51, 52, 53, 53], [52, 44, 45, 45], [33, 35, 36, 36]]),  # robust
]
# type of scaling needed
info_list = ["robust", "standard", "categorical", "categorical", "standard", "robust"]

# Flatten the 2D arrays into individual columns
flattened_data = np.hstack([data.reshape(-1, 1) for data in data_list])

# Define indices for each type
robust_indices = [0, 5]
standard_indices = [1, 4]
categorical_indices = [2, 3]

# Create transformers for each type
robust_transformer = Pipeline(steps=[("robust_scaler", RobustScaler())])
standard_transformer = Pipeline(steps=[("standard_scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder())])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("robust", robust_transformer, robust_indices),
        ("standard", standard_transformer, standard_indices),
        ("categorical", categorical_transformer, categorical_indices),
    ]
)
# Add AgglomerativeClustering step to the pipeline
clustering = AgglomerativeClustering(n_clusters=3)

# Create and apply the pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("rescale_all", RescaleAllToCommonRange()), ("clustering", clustering)],
    verbose=True,
)

# Fit and transform the data
fitted = pipeline.fit(flattened_data)
final_data = pipeline.fit_predict(flattened_data)

preprocessed_data = pipeline.named_steps["preprocessor"].transform(flattened_data)
print(preprocessed_data)

rescaled_data = pipeline.named_steps["rescale_all"].transform(preprocessed_data)
print(rescaled_data)

#  Reshape the processed data back into the original 2D array shapes
height, width = data_list[0].shape
reshaped_data_list = []

start_idx = 0
for data in data_list:
    num_elements = data.size
    reshaped_data = final_data[start_idx : start_idx + num_elements].reshape(data.shape)
    reshaped_data_list.append(reshaped_data)
    start_idx += num_elements

for data in reshaped_data_list:
    print(data)
