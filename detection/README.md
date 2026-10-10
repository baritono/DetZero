# DetZero - Detection Module


## Intro
- This is the detection module of DetZero framework. It is very similiar to the usage of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), but only contains one detector, CenterPoint.

- We provide all the versions of our ensembled models' configs, one of the prerequisite is to generate multi-frame information and ground-truth database. Thanks to our flexible structure design of information, we only preprocess the raw data for only one time (refer to [Data Preprocess](../docs/DATA_PREPROCESS.md)).


## Data Preprocess
Please see [Data Preprocess](../docs/DATA_PREPROCESS.md) and make sure the dataset are placed at the required file folder.


## Running
- a. compile the module
```shell
cd DetZero/detection &&
python setup.py develop
```

- b. train one model
```shell
cd DetZero/detection/tools &&
python train.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml
```

- c. infer with the best model
```shell
cd DetZero/detection/tools &&
python test.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --ckpt <PATH_TO_CKPT>
```

- d. infer with the best model TTA version
```shell
cd DetZero/detection/tools &&
python test.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --ckpt <PATH_TO_CKPT> --set DATA_CONFIG.TTA True
```



## Tips
- If you want to change the value of some common settings, please use `--set` following your training command rather than modify the yaml file straightly. For example, `--set OPTIMIZATION.BATCH_SIZE_PER_GPU 16` for changing batch size.

- If you want to use some tag to mark this specific experiment, please use `--extra_tag` following your command. For example, `--extra_tag bs16` for batch size 16 experiment.

- Training always writes TensorBoard logs to `output/<cfg path>/<extra_tag>/tensorboard`. To also log to [Weights & Biases](https://wandb.ai), `pip install "wandb<0.26"`, log in (`wandb login` or `WANDB_API_KEY`), and add `--wandb` (optionally `--wandb_project`, `--wandb_entity`, `--wandb_name`, `--wandb_tags`). Every TensorBoard scalar is mirrored to W&B, together with the config and GPU utilization. The same flags work for `refining/tools/train.py`.
