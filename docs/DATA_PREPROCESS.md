# Data Preprocess

## Dataset Preparation
Currently we only provide the processing of Waymo dataset. 

- place (or soft-link) the raw tfrecords of **all splits** together in one flat `raw_data` folder under the
  repository-level `data/waymo` (the split membership comes from `ImageSets/*.txt`):
	```
	DetZero
	├── data
	│   ├── waymo
	│   │   │── ImageSets
	│   │   │── raw_data
	│   │   │   │── segment-xxxxxxxx_with_camera_labels.tfrecord
	│   │   │   │── ....
	├── detection
	```
	For example, with the official `training/`, `validation/`, `testing/` download folders:
	```shell
	mkdir -p data/waymo/raw_data
	ln -s <WAYMO_ROOT>/{training,validation,testing}/*.tfrecord data/waymo/raw_data/
	```

- process waymo infos:
  ```shell
  cd detection
  python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml --func create_waymo_infos
  ```
  Optional flags: `--splits train val` to skip the test split, and `--workers N` to limit how many
  sequences are processed concurrently (each worker holds a whole tfrecord in memory, about 3 GB at peak;
  the default is one per CPU core). The point clouds of train + val take about 830 GB, test another ~125 GB.
  Sequences that already have a `.pkl` are skipped, so an interrupted run can simply be restarted.

- generate database for gt-sampling
  ```
  cd detection
  python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml --func create_waymo_database
  ```
  The multi-sweep detector configs use their own databases, so repeat this with `waymo_3sweeps.yaml`
  and `waymo_5sweeps.yaml` if you train those models. This step needs a CUDA GPU.

### NOTE
We have provided a flexible data info structure and processing logic to satisfy single-frame or multi-frame (e.g., 2, 3, 5, ...) point clouds loading without further repeated pre-processings.
```python
    def get_sweep_idxs(current_info, sweep_count=[0, 0], current_idx=0):

        assert type(sweep_count) is list and len(sweep_count) == 2,\
            "Please give the upper and lower range of frames you want to process!"

        current_sample_idx = current_info["sample_idx"]
        current_seq_len = current_info["sequence_len"]

        target_sweep_list = np.array(list(range(sweep_count[0], sweep_count[1]+1)))
        target_sample_list = current_sample_idx + target_sweep_list
        # set the low and high thresh to extract multi frames in current sequence
        target_sample_list = [i if i >= 0 else 0 for i in target_sample_list]
        target_sample_list = [i if i < current_seq_len else current_seq_len-1 for i in target_sample_list]
        # get the index of target frames in the waymo info list
        target_idx_res = np.array(target_sample_list) - current_sample_idx
        target_idx_list = current_idx + target_idx_res

        return target_idx_list
```

