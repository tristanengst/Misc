# Miscellaneous utilities
Utility functions for deep learning research, primarily in computer vision.

### LMDB Datasets
These are useful in settings where you can't have many files, like ComputeCanada clusters. If `IMAGEFOLDER` is a folder that can be read into a TorchVision `ImageFolder`, then
```
python -c "import Misc; Misc.image_folder_to_lmdb(IMAGEFOLDER, LMDBFILE)"
```
generates a file named `LMDBFILE`. Combined with such files, the `LMDBImageFolder` is a drop-in replacement for a TorchVision `ImageFolder`:
```
dataset = torchvision.datasets.ImageFolder(IMAGEFOLDER, transforms=...)
dataset = Misc.LMDBImageFolder(LMDBFILE, transforms=...)
```
On modern SSDs with many workers, loading data from one isn't appreciably faster than from the other.