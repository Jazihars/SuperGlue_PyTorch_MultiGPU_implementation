# SuperGlue_PyTorch_MultiGPU_implementation

This Python script is based on [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch), but inplement the Multi-GPU training with [PyTorch DistributedDataParallel](https://pytorch.org/docs/1.8.0/generated/torch.nn.parallel.DistributedDataParallel.htmls)。

## Environment prepareing
In my Linux termiral, run this command:
``` bash
cat /etc/redhat-release
```
we can see this:
```
CentOS Linux release 7.5.1804 (Core)
```
So my experiments run on a `CentOS Linux` machine.
Run this command:
``` bash
nvcc -V
```
we can see:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:10:02_PDT_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.TC455_06.29069683_0
```
So my cuda version is 11.1

First, let's clone [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch). Run this command:
``` bash
git clone https://github.com/HeatherJiaZG/SuperGlue-pytorch.git
cd SuperGlue-pytorch
```
（If you clone [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch) from github is too slow, you can use [Gitee](https://gitee.com/) to clone.）
Then, let's create a new conda environment:
``` bash
conda create --name superglue python=3.8
conda activate superglue
```
Then, use these commands to build the environment:
``` bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1
pip install opencv-python
pip install matplotlib
pip install black
pip install tqdm
pip install scipy
```
Again, use command `pip list` to see all my package's version:
```
Package           Version
----------------- --------
black             22.1.0
click             8.0.3
cycler            0.11.0
fonttools         4.29.1
kiwisolver        1.3.2
matplotlib        3.5.1
mkl-fft           1.3.1
mkl-random        1.2.2
mkl-service       2.4.0
mypy-extensions   0.4.3
numpy             1.21.2
olefile           0.46
opencv-python     4.5.5.62
packaging         21.3
pathspec          0.9.0
Pillow            8.4.0
pip               22.0.3
platformdirs      2.5.0
pyparsing         3.0.7
python-dateutil   2.8.2
scipy             1.8.0
setuptools        60.8.2
six               1.16.0
tomli             2.0.1
torch             1.8.0
torchvision       0.9.0
tqdm              4.62.3
typing_extensions 4.0.1
wheel             0.37.1
```
My `cuda` version is `11.1`, my `torch` version is `1.8.0`, my `torchvision` version is `0.9.0`, my `opencv-python` version is `4.5.5.62`
Then change `/SuperGlue-pytorch/load_data.py` `class SparseDataset(Dataset): def __init__(self, train_path, nfeatures):`'s:
``` python
self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
```
to
``` python
self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
```
change `/SuperGlue-pytorch/load_data.py` `class SparseDataset(Dataset): def __getitem__(self, idx):`'s
``` python
if len(kp1) < 1 or len(kp2) < 1:
```
to
``` python
if len(kp1) <= 1 or len(kp2) <= 1:
```
Then use command
``` bash
touch train_DistributedDataParallel.py
```
to create a new python script. The script's path is `/SuperGlue-pytorch/train_DistributedDataParallel.py`.Copy all code of [my script](https://github.com/Jazihars/SuperGlue_PyTorch_MultiGPU_implementation/blob/main/train_DistributedDataParallel.py) into my own script `/SuperGlue-pytorch/train_DistributedDataParallel.py`.Change the data path in `/SuperGlue-pytorch/train_DistributedDataParallel.py` to this:
``` python
parser.add_argument(
    "--train_path",
    type=str,
    default="/data/XXX/coco2014/train2014/",
    help="Path to the directory of training imgs.",
)
```
Warning: `default="/data/XXX/coco2014/train2014/"` must have a `/` in the end.

Run this command to start train:
``` bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_DistributedDataParallel.py
```
Then you can train SuperGlue in Multi-GPU mode.

I haven't used apex to implementation multi-GPU training. Maybe I will implement the apex implementation in the future.

This code is under MIT License.

Thank you for [HeatherJiaZG SuperGlue-pytorch](https://github.com/HeatherJiaZG/SuperGlue-pytorch). My script is based on HeatherJiaZG's code.