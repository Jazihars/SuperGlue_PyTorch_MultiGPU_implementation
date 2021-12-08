# SuperGlue_PyTorch_MultiGPU_implementation

This Python script is based on [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch), but inplement the Multi-GPU training with [PyTorch DistributedDataParallel](https://pytorch.org/docs/1.8.0/generated/torch.nn.parallel.DistributedDataParallel.htmls)。

If you want to run this script, please clone [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch) and use this script instead of original HeatherJiaZG SuperGlue-pytorch train.py。

The environment is based on this(Use the command `pip list` to see, some package's version is different with HeatherJiaZG's SuperGlue-pytorch):
```
Package           Version
----------------- --------
apex              0.1
black             21.12b0
click             8.0.3
cycler            0.11.0
fonttools         4.28.3
kiwisolver        1.3.2
matplotlib        3.5.0
mkl-fft           1.3.1
mkl-random        1.2.2
mkl-service       2.4.0
mypy-extensions   0.4.3
numpy             1.21.2
olefile           0.46
opencv-python     4.5.4.60
packaging         21.3
pathspec          0.9.0
Pillow            8.4.0
pip               21.3.1
platformdirs      2.4.0
pyparsing         3.0.6
python-dateutil   2.8.2
scipy             1.7.3
setuptools        59.4.0
setuptools-scm    6.3.2
six               1.16.0
tomli             1.2.2
torch             1.8.0
torchvision       0.9.0
tqdm              4.62.3
typing_extensions 4.0.1
wheel             0.37.0
```
My cuda version is 11.1

I haven't used apex implementation. Maybe I will implement the apex implementation in the future.

This code is under MIT License.

Thank you for [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch). My script is based on HeatherJiaZG's code.