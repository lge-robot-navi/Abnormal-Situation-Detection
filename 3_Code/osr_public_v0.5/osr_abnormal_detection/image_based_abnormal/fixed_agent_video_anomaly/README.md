## Requirements
* python 3
* pip install -r requirements.txt

## Download dataset
```
$ mkdir dataset
$ cd dataset
$ wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
$ tar -xzf UCSD_Anomaly_Dataset.tar.gz
```

## Usage

### Prepare data
```
$ python preprocess_data.py
```
### Train
```
$ python train.py
```
**Train Autoencoder**
```
$ python train.py --model AE
```
**Train Adversarial Autoencoder**
```
$ python train.py --model AAE
```
####Arguments
```
$ python test.py -h
	usage: train.py [-h] [--dataset DATASET] [--save_dir SAVE_DIR] [--model MODEL]
				   [--epochs EPOCHS] [--batch_size BATCH_SIZE]
				   [--patch_height PATCH_HEIGHT] [--patch_width PATCH_WIDTH]
				   [--strides STRIDES]
	optional arguments:
	  -h, --help            show this help message and exit
	  --dataset DATASET
	  --save_dir SAVE_DIR
	  --model MODEL
	  --epochs EPOCHS
	  --batch_size BATCH_SIZE
	  --patch_height PATCH_HEIGHT
	  --patch_width PATCH_WIDTH
	  --strides STRIDES
```

* `--dataset`: File name of input images *Default*: `UCSDPed2`
* `--save_dir`: Path of saving preprocessed images *Default*: `save`
* `--learn_rate`: Model *Default*: `AAE`
* `--num_epochs`: Epochs *Default*: `30`
* `--batch_size`: Batch size *Default*: `128`
* `--patch_height`: Patch height *Default*: `45`
* `--patch_width`: Patch width *Default*: `45`
* `--strides`: strides *Default*: `25`

### Test
```
$ python test.py
```
