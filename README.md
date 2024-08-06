##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset](#dataset)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)
5. [Acknowledgement](#acknowledgement)
6. [Citation](#citation)

# SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models (CVPR'24)
This repository provides the official PyTorch implementation of the following paper: 
<div align="center">
  <a href="https://arxiv.org/abs/2312.07865" target="_blank">SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models <br />
  <a href="http://home.ustc.edu.cn/~wangfeifei/" target="_blank">Feifei&nbsp;Wang</a><sup>1,2,*</sup> &emsp;
  <a href="https://scholar.google.com/citations?user=VCX7itEAAAAJ" target="_blank">Zhentao&nbsp;Tan</a><sup>2,1</sup> &emsp;
  <a href="https://scholar.google.com/citations?hl=en&user=-wfXmM4AAAAJ" target="_blank">Tianyi&nbsp;Wei</a><sup>1</sup> &emsp;
  <a href="https://scholar.google.com/citations?user=srajsjoAAAAJ&hl=en" target="_blank">Yue&nbsp;Wu</a><sup>2</sup>&emsp;
  <a href="https://shikiw.github.io/" target="_blank">Qidong&nbsp;Huang</a><sup>1</sup>&emsp;
  <br> <br>
  
  
 <br><sup>1</sup>University of Science and Technology of China, <sup>2</sup>Alibaba Cloud<br>
</div>
<br>


## Environment setup
Install dependencies:
```shell
cd SimAC
conda create -n simac python=3.9  
conda activate simac
pip install -r requirements.txt  
```

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
</table>

Please download the pretrain weights and define "$MODEL_PATH" in the script. Note: Stable Diffusion version 2.1 is the default version in all of our experiments.

> GPU allocation: All experiments are performed on a single NVIDIA 48GB A6000 GPU.

## Dataset 
Thanks for Anti-Dreambooth's great efforts, there are two datasets: VGGFace2 and CelebA-HQ which are provided at [here](https://drive.google.com/drive/folders/1vlpmoKPZVgZZp-ANBzg915hOWPlCYv95?usp=sharing).

For convenient testing, we have provided a split set of one subject in CelebA-HQ at `./data/CelebA-HQ/103` as the Anti-dreambooth does.

## How to run

To defense Stable Diffusion version 2.1 (default) with ASPL, you can run
```
bash scripts/attack_aspl.sh
```

To defense Stable Diffusion version 2.1 (default) with SimAC, you can run
```
bash scripts/attack_timesteps.sh
```


If you want to train a DreamBooth model from your own data, whether it is clean or perturbed, you may run the following script:
```
bash scripts/train_dreambooth_alone.sh
```

Inference: generates examples with multiple-prompts
```
python infer.py --model_path <path to DREAMBOOTH model>/checkpoint-1000 --output_dir $<path to DREAMBOOTH model>/checkpoint-1000-test-infer
```

## Contacts
If you have any problems, please open an issue in this repository or send an email to [wangfeifei@mail.ustc.edu.cn](mailto:wangfeifei@mail.ustc.edu.cn).


## Acknowledgement
This repo is heavil based on [Anti-DB](https://github.com/VinAIResearch/Anti-DreamBooth). Thanks for their impressive works!

## Citation
Details of algorithms and experimental results can be found in [our following paper](https://arxiv.org/abs/2312.07865):
```bibtex
@inproceedings{wang2024simac,
  title={SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models},
  author={Feifei Wang and Zhentao Tan and Tianyi Wei and Yue Yue and Qidong Huang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12047--12056},
  year={2024}
}
```
**Please CITE** our paper if you find this work useful for your research.
