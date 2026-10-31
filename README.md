# SGPDFuse

Haozhen Wei, Chengjun Jiang, Yutong Guo, Xinrui Ju, Xingyuan Li, Xiang Chen, Jinyuan Liu, **"SGPDFuse: Semantically-Guided Physics-Disentanglement General Multi-Modal Image Fusion"**, Proceedings of the 34th ACM International Conference on Multimedia **(ACM MM)**, 2026.

```
@inproceedings{wei2026sgpdfuse,
  title={SGPDFuse: Semantically-Guided Physics-Disentanglement General Multi-Modal Image Fusion},
  author={Wei, Haozhen and Jiang, Chengjun and Guo, Yutong and Ju, Xinrui and Li, Xingyuan and Chen, Xiang and Liu, Jinyuan},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## Environment

```
# create virtual environment
conda create -n SGPDFuse python=3.10
conda activate SGPDFuse
# install requirements
pip install -r requirements.txt
# install DINOv3
pip install --no-deps git+https://github.com/facebookresearch/dinov3.git
```

If GitHub is not reachable through pip, clone [DINOv3](https://github.com/facebookresearch/dinov3) manually and run `pip install --no-deps -e dinov3`.

## Datasets

Paired images share the same file name in two folders:

```
datasets/
  M3FD/  {ir, vi}/
  MFIF/  {near, far}/
  MEFB/  {low, high}/
```

## Test Image Fusion

Pretrained weights can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1HWhbChDLMckujipe6N-sMg?pwd=rtuv) (code: rtuv). Put them in **"ckpt/"**, then you can test our fusion method through

```
python test.py --config config/IVIF.yml --ckpt ckpt/IVIF.pth --data datasets/M3FD/test --out results/IVIF
python test.py --config config/MFF.yml --ckpt ckpt/MFF.pth --data datasets/MFIF/test --out results/MFF
python test.py --config config/MEF.yml --ckpt ckpt/MEF.pth --data datasets/MEFB/test --out results/MEF
```

Add `--eps` to save the predicted emissivity maps.

## Train

Download the DINOv3 weights from [DINOv3](https://github.com/facebookresearch/dinov3) and put them in **"ckpt/"**. Then

```
python train.py --config config/IVIF.yml
torchrun --nproc_per_node=8 train.py --config config/IVIF.yml
```

Training runs reconstruction (stage 1) and fusion (stage 2) in sequence. All options are set in `config/*.yml`.
