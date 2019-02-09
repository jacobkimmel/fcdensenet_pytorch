# PyTorch Tiramisu

A [PyTorch](https://pytorch.org) implementation of Fully Convolutional DenseNets for semantic segmentation, as described in [The One Hundred Layer Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326) by Jegou *et. al.*.

## Usage

The CLI provided in `main.py` offers entry points for model training and mask prediction. The CLI uses a configuration file to input most of the required parameters. Parameters are described in 

```bash
python main.py -h
```

and the configuration file can use any format supported by [`configargparse`](https://github.com/bw2/ConfigArgParse).

An example configuration is provided in `default_config.txt`.

Required parameters must be passed either in the configuration file or on the command line. Command line arguments will supercede corresponding settings in the configuration file.

### Training

```bash
python main.py --command train --config $PATH_TO_CONFIG_FILE
```

### Prediction

```bash
python main.py --command predict --config $PATH_TO_CONFIG_FILE
```

## Development

This tool was originally a product of the [Laboratory of Cell Geometry](https://cellgeometry.ucsf.edu/) at the [University of California, San Francisco](https://ucsf.edu).