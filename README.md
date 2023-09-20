# <p align=center>`DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation`</p>

> **Authors:**
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Xuying Zhang](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=zh-CN),
> [Li Liu](https://scholar.google.com/citations?hl=zh-CN&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)


This official repository contains the source code, pre-trained, trained checkpoints, and evaluation toolbox of paper 'DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation'. The technical report could be found at [arXiv](https://arxiv.org/pdf/2309.09668.pdf). 
The code for pre-training and RGB-D saliency will be released soon.


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=dformer-rethinking-rgbd-representation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-des)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-des?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-stere)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-stere?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-sip)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-sip?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-nlpr)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nlpr?p=dformer-rethinking-rgbd-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dformer-rethinking-rgbd-representation/rgb-d-salient-object-detection-on-nju2k)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nju2k?p=dformer-rethinking-rgbd-representation)

<p align="center">
    <img src="figs/DFormer.png" width="600"  width="1200"/> <br />
    <em> 
    Figure 1: Comparisons between the existing methods and our DFormer (RGB-D Pre-training).
    </em>
</p>

<p align="center">
    <img src="figs/overview.jpg" width="600"  width="1200"/> <br />
    <em> 
    Figure 2: Overview of the DFormer.
    </em>
</p>






## 1. 🌟  NEWS 

- [2023/09/05] Releasing the codebase of DFormer and all the pre-trained checkpoints.

> We invite all to contribute in making it more acessible and useful. If you have any questions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn). If you are using our code and evaluation toolbox for your research, please cite this paper ([BibTeX]()).



## 2. 🚀 Get Start

**0. Install**

```
conda create -n dformer python=3.10 -y
conda activate dformer
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install tqdm opencv-python scipy tensorboardX tabulate easydict
```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

By default, you can put datasets into the folder 'datasets' or use 'ln -s path_to_data datasets'.

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|
- **Checkpoints:** 

ImageNet-1K Pre-trained DFormers T/S/B/L can be downloaded at 

| Pre-trained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|:---: |:---:|:---:|:---:|



NYUDepth v2 trained DFormers T/S/B/L can be downloaded at 

| NYUDepth v2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|:---: |:---:|:---:|:---:|


*SUNRGBD 

| SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 
|:---: |:---:|:---:|:---:|


 <br />

Orgnize the checkpoints and dataset folder in the following structure:


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



 <br /> 




**2. Train.**

You can change the `local_config' files in the script to choose the model for training. 
```
bash train.sh
```


**3. Eval.**

You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 
```
bash eval.sh
```



## 🚩 Performance

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p>

<p align="center">
    <img src="figs/Sal.jpg" width="600"  width="1200"/> <br />
    <em> 
    </em>
</p>

## 🕙 ToDo
- [ ] Release the code of RGB-D pre-training.
- [ ] Release the DFormer code for RGB-D salient obejct detection.

> We invite all to contribute in making it more acessible and useful. If you have any questions or suggestions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn) or raise an issue. 


## Reference
You may want to cite:
```
@article{yin2023dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation}, 
  author={Bowen Yin and Xuying Zhang and Zhongyu Li and Li Liu and Ming-Ming Cheng and Qibin Hou},
  journal={arXiv preprint arXiv:2309.09668},
  year={2023}
}
```


### Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [CMNext](https://github.com/jamycheung/DELIVER). Thanks for their authors.



### License

Code in this repo is for non-commercial use only.






