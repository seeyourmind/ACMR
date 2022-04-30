# ACMR

Original PyTorch implementation of "Learning Aligned Cross-Modal Representations for Generalized Zero-Shot Classification".

<img src="https://github.com/seeyourmind/ACMR/blob/master/arch-ACMR.png" style="zoom:80%;" />

### Requirements

The code was implemented using Python 3.6.0 and trained on one NIVIDIA GeForce GTX TITAN X GPU . 

The following packages:

```
torch==1.5.0
numpy==1.19.4
scipy==1.2.1
tqdm==4.54.1
scikit_learn==0.23.2
```

### Data

Download the following [CADA]( https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0) and put it in this repository. Next to the folder "model", there should be a folder "data".

### Citation

If this work is helpful for you, please cite our paper.

```bibtex
@inproceedings{FangZY2022ACMR,
  author    = {Fang, Zhiyu and 
               Zhu, Xiaobin and 
               Yang, Chun and 
               Han, Zheng and 
               Qin, Jingyan and 
               Yin, Xu-Chengg},
  title     = {Learning Aligned Cross-Modal Representation for Generalized Zero-Shot 
               Classification},
  booktitle = {36th AAAI Conference on Artificial Intelligence},
  year      = {2022},
}
```

### Ackowledgement

We thank the [CADA_VAE](https://github.com/edgarschnfld/CADA-VAE-PyTorch) repos providing helpful components in our work.

