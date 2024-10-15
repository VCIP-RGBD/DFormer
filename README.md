# <p align=center>`DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation (ICLR 2024)`</p>

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=dformer-rethinking-rgbd-representation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-des)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-des?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-stere)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-stere?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-sip)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-sip?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-nlpr)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nlpr?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-nju2k)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nju2k?p=dformer-rethinking-rgbd-representation) -->


> **Authors:**
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Xuying Zhang](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=zh-CN),
> [Li Liu](https://scholar.google.com/citations?hl=zh-CN&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)


[Paper Link](https://arxiv.org/abs/2309.09668) | 
[Homepage](https://yinbow.github.io/Projects/DFormer/index.html) |
[公众号解读(集智书童)](https://mp.weixin.qq.com/s/lLFejycBr8o7JNoirRDmjQ) |
[DFormer-SOD](https://github.com/VCIP-RGBD/DFormer-SOD) |


:robot:[RGBD-Pretrain(You can train your own encoders)](https://github.com/VCIP-RGBD/RGBD-Pretrain)

:anchor:[Application to new datasets(添加新数据集)](https://github.com/VCIP-RGBD/DFormer/tree/main/figs/application_new_dataset)



<!-- This official repository contains the source code, pre-trained, trained checkpoints, and evaluation toolbox of paper 'DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation'. The technical report could be found at [arXiv](https://arxiv.org/pdf/2309.09668.pdf). 
The code for pre-training and RGB-D saliency will be released soon. -->
This official repository of 'DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation'.
We provide the RGBD pretraining code in [RGBD-Pretrain](https://github.com/VCIP-RGBD/RGBD-Pretrain).
You can pretrain more powerful RGBD encoders and contribute to the RGBD research.

We invite all to contribute in making it more acessible and useful. If you have any questions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn). If you are using our code and evaluation toolbox for your research, please cite this paper ([BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:GdonbkKZMYsJ:scholar.google.com/&output=citation&scisdr=ClEqKQU5EL_6hIbkmOc:AFWwaeYAAAAAZQvigOeM_E2bhS0d1niD6tYkedk&scisig=AFWwaeYAAAAAZQvigF3P1qyHXOMhOEt-zalsD8w&scisf=4&ct=citation&cd=-1&hl=zh-CN)).



<p align="center">
    <img src="figs/DFormer.png" width="600"  width="1200"/> <br />
    <em> 
    Figure 1: Comparisons between the existing methods and our DFormer (RGB-D Pre-training).
    </em>
</p>

<!-- <p align="center">
    <img src="figs/overview.jpg" width="600"  width="1200"/> <br />
    <em> 
    Figure 2: Overview of the DFormer.
    </em>
</p> -->






## 1. 🌟  NEWS 

<!-- - [2023/09/05] Releasing the codebase of DFormer and all the pre-trained checkpoints.

- [2023/10/26] Releasing the RGBD SOD codebase of DFormer at [DFormer-SOD](https://github.com/VCIP-RGBD/DFormer-SOD).

- [2023/12/03] Adding the tutorial about adding new datasets at [Application to new datasets(添加新数据集)](https://github.com/VCIP-RGBD/DFormer/tree/main/figs/application_new_dataset). -->

- [2024/10/12] Based on our DFormer, Wu's method UBCRCL has won the RUNNER-up at [Endoscopic Vision Challenge SegSTRONG-C Subchallenge](https://segstrongc.cs.jhu.edu/) of MICCAI 24. Congratulation!
- [2024/01/16] Our DFormer has been accpeted by The International Conference on Learning Representations (ICLR 2024).
- [2024/04/21] We have upgraded and optimized the framework, greatly reducing training time, i.e., training duration for DFormer-L is reduced to ~12h from over 1day.


## 2. 🚀 Get Start

**0. Install**

```bash
conda create -n dformer python=3.10 -y
conda activate dformer

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

By default, you can put datasets into the folder 'datasets' or use 'ln -s path_to_data datasets'.

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|

Compred to the original datasets, we map the depth (.npy) to .png via 'plt.imsave(save_path, np.load(depth), cmap='Greys_r')', reorganize the file path to a clear format, and add the split files (.txt).



- **Checkpoints:** 

ImageNet-1K Pre-trained DFormers T/S/B/L and NYUDepth or SUNRGBD trained DFormers T/S/B/L can be downloaded at:
<!-- 
| Pre-trained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|:---: |:---:|:---:|:---:|




NYUDepth v2 trained DFormers T/S/B/L can be downloaded at 

| NYUDepth v2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|:---: |:---:|:---:|:---:|


*SUNRGBD 

| SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 
|:---: |:---:|:---:|:---:| -->


| Weights | GoogleDrive | OneDrive | BaiduNetdisk|
|-------|-------| - | - |
| Pretrained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|NYUDepthv2 (57.2mIoU)|[GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|SUNRGBD (52.5mIoU)|[GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 


 <br />


<details>
<summary>Orgnize the checkpoints and dataset folder in the following structure:</summary>
<pre><code>

```shell
<checkpoints>
|-- <pretrained>
    |-- <DFormer_Large.pth.tar>
    |-- <DFormer_Base.pth.tar>
    |-- <DFormer_Small.pth.tar>
    |-- <DFormer_Tiny.pth.tar>
|-- <trained>
    |-- <NYUDepthv2>
        |-- ...
    |-- <SUNRGBD>
        |-- ...
<datasets>
|-- <DatasetName1>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <Depth>
        |-- <name1>.<DepthFormat>
        |-- <name2>.<DepthFormat>
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
```

</code></pre>
</details>




 <br /> 




**2. Train.**

You can change the `local_config' files in the script to choose the model for training. 
```
bash train.sh
```

After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX is depends on the training config.


**3. Eval.**

You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 
```
bash eval.sh
```

**4. Visualize.**

```
bash infer.sh
```

**5. FLOPs & Parameters.**

```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python benchmark.py --config local_configs.NYUDepthv2.DFormer_Large
```

**6. Latency.**

```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Large
```

ps: The latency highly depends on the devices. It is recommended to compare the latency on the same devices. 


## 🚩 Performance

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p>

<!-- <p align="center">
    <img src="figs/Sal.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p> -->

## 🕙 ToDo
- [ ] Tutorial on applying the DFormer encoder to the frameworks of other tasks
- ~~[-] Release the code of RGB-D pre-training.~~
- ~~[-] Tutorial on applying to a new dataset.~~
- ~~[-] Release the DFormer code for RGB-D salient obejct detection.~~

> We invite all to contribute in making it more acessible and useful. If you have any questions or suggestions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn) or raise an issue. 


## Reference
You may want to cite:
```
@article{yin2023dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhongyu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2309.09668},
  year={2023}
}
```


### Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [CMNext](https://github.com/jamycheung/DELIVER). Thanks for their authors.



### License

Code in this repo is for non-commercial use only.






