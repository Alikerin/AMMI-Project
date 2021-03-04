# Color Transfer with Differentiable Histogram
This project is based on [Differentiable Histogram with Hard-Binning](https://arxiv.org/pdf/2012.06311.pdf).


## Prerequisites
- Python 3.6.x
- [PyTorch 1.6.0 & torchvision 0.7.0](http://pytorch.org/)


## Dataset
Dataset was generated from edges2shoes dataset using `create_dataset.py`
```bash
python create_dataset.py --original_dir "edges2shoes" --new_dir "edges2shoes_triplets" --n_repeat 1
```


## Training

### Train model

```bash
python main.py --mode train --data_dir [data_directory] --out_dir [output_directory] --n_epoch 100 --resize 143 --crop 128 --batch_size 30 --hist_loss mae --lr_decay_start 30 --lr_decay_n 70
```
### Resume training from checkpoint
```bash
python main.py --mode train --data_dir [data_directory] --out_dir [output_directory] --n_epoch 100 --resize 143 --crop 128 --batch_size 30 --hist_loss mae --lr_decay_start 30 --lr_decay_n 70 --pretrain_path ./[output_directory]/xxx/xxx.pt
```

### Plot loss stats from train.json
```bash
python plot.py --dir [output_directory]
```
It will look for train.json in the directory and output plots as result.png.

### See more options available
```bash
python main.py -h
```

## Testing
```bash
python main.py --mode test --crop 128 --resize 143 --data_dir datasets/edges2shoes --pretrain_path [output_directory]/xxx.pt
```
This generates all images from test set and save them to ./checkpoints/xxx/images/test/.
