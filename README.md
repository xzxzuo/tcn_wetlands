## Function of each file
- [`tcn.py`](./tcn.py) The model is built upon the [TCN architecture](https://github.com/locuslab/TCN.git). Here you can find what `TCN` block is.

- [`model.py`](./model.py) Create a TCN-based model which accepts a time series of pixels and outputs **prediction** of the next time step and **feature** of each pixel.

- [`dataset.py`](./dataset.py) Create dataset for training.

- [`utils.py`](./utils.py) Reusable functions. 

- [`train_predict.py`](./train_predict.py) Train a TCN-based pixel-wise model and use next step prediction as loss.

- [`extract_feature_mem.py`](./extract_feature_mem.py) Load a trained checkpoint and extract pixel-level temporal embeddings for all pixels. It uses memory-mapped arrays to save features directly to disk, which helps avoid RAM OOM when processing large regions such as Örebro län.

- [`cluster_features_memmap.py`](./cluster_features_memmap.py) Cluster the extracted pixel embeddings using `MiniBatchKMeans`. It reads `features.npy`, `coords.npy`, and `metadata.npz` from the feature extraction output folder, fits `KMeans` in batches, predicts a cluster label for each valid pixel, and writes the result back to a spatial cluster map.

## Usage
### train_predict.py
Run the following script: 
```shell
python train_predict.py \ 
    --images /path/to/sar_images \ 
    --output-dir /path/to/trained_model \ 
    --channels 64,64,64 \ 
    --kernel-size 2 \ 
    --epochs 20 \ 
    --batch-size 16384 \ 
    --lr 1e-3 \ 
    --num-workers 8 \ 
    --use-recency-input \ 
    --half-life-days 6 \ 
    --use-wandb
```
after training, you'll get:

- best_tcn_encoder.pt &emsp; # Best model checkpoint based on validation loss 
- training_metadata.json &emsp; # Training configuration and dataset metadata 
- training_history.json &emsp; # Train/validation loss history 
- tile_split_coords.npz &emsp; # Spatial train/validation pixel split 
- test_metrics.json &emsp; # Best validation loss summary

#### Explanation of args
- `--channels` Separated by commas, e.g., `64,64,64` will generate **3** TCN blocks, the hidden channels and the output embedding dimension are **64**.

- `--use-recency-input` The input is `[B, T, 2]` if using this argument, otherwise, the input is `[B, T, 1]`. `[B, T, 1]` only contains raw pixel value, while `[B, T, 2]` contains both raw pixel and weighted value. 

### extract_feature_mem.py
Run the following script:
```shell
python extract_feature_mem.py \ 
    --images /path/to/test_sar_images \ 
    --checkpoint /path/to/trained_model/best_tcn_encoder.pt \ 
    --output-dir /path/to/extracted_features \ 
    --feature-mode last \ 
    --batch-size 4096 \ 
    --num-workers 2 \ 
    --device cuda \ 
    --use-recency-input \ 
    --half-life-days 6
```
you'll get:
- features.npy &emsp; # Pixel embeddings, shape [N, C] 
- coords.npy &emsp; # Pixel coordinates, shape [N, 2], stored as [row, col] 
- metadata.npz &emsp; # Image size, reference path, feature dimension, recency settings

If dates cannot be parsed from filenames, provide them manually:
```shell
python extract_feature_mem.py \ 
    --images img1.tif img2.tif img3.tif img4.tif img5.tif \ 
    --checkpoint /path/to/trained_model/best_tcn_encoder.pt \ 
    --output-dir /path/to/extracted_features \ 
    --feature-mode last \ 
    --use-recency-input \ 
    --half-life-days 6 \ 
    --recency-dates 2020-08-04 2020-08-10 2020-08-16 2020-08-22 2020-08-28
```
#### Explanation of args
`--feature-mode` This argument could be selected from `["mean", "last", "time"]`. 
- `mean` refers to calculating the average of all features extracted at each time point. 
- `last` refers to extracting the last time features from test input. 
- `time` refers to extracting a specific time point feature, and if you use `time`, you should add another argument `--time-index`, e.g., `--time-index 2` refers to the second time point, if the input date is [08-04, 08-10, 08-16, 08-22, 08-28], it extracts **08-16** features (only see [08-04, 08-10, 08-16]).

### cluster_features_memmap.py
Run the following script:
```shell
python cluster_features_memmap.py \ 
    --feature-dir /path/to/extracted_features \ 
    --output-dir /path/to/clustered_output \ 
    --n-clusters 8 \ 
    --batch-size 8192 \ 
    --fit-epochs 2 \ 
    --standardize \
    --save-tif 
```
you'll get:
- preview_k{K}_h{H}_d{D}.png &emsp; # Downsampled cluster preview image 
- cluster_map_k{K}_h{H}_d{D}.tif &emsp; # Georeferenced cluster map, if --save-tif is used

note: **k** for number of clusters, **h** for half-life days, or -1 if recency input is not used, and **d** for feature embedding dimensions.

`--standardize`: Optional flag for feature normalization before KMeans clustering. 
When enabled, each embedding dimension is standardized using the mean and standard deviation computed over all valid pixels. 
This prevents high-variance feature dimensions from dominating the Euclidean distance used by KMeans.

