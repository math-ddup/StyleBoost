# StyleBoost
**"StyleBoost: Controlling Style-Content Fusion with SVD for Text-Driven Generation"**
---
## 📖 Introduction
This repository provides the official implementation of **StyleBoost**, a text-driven style transfer framework based on diffusion models.  
We propose a novel **SVD-based control mechanism** to balance content preservation and style adaptation.

---

## 📖 Mehthod
![Framework](/images/result/0.jpg)


## Results
![Visualization Results](/images/result/2.png)

![More Results](/images/result/3.png)
---

## 1.Download
```bash
# git clone this repository
git clone https://github.com/math-ddup/StyleBoost.git
cd StyleBoost

# download the models
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```

## 2.Set Up the Environment
```bash
conda create -n styleboost python=3.10 -y
conda activate styleboost

pip install -r requirements.txt
```
## 3. Run Inference
```bash
python infer_style.py
```


## Acknowledgements
Our work is mainly based on the following projects:
- [InstantStyle](https://github.com/instantX-research/InstantStyle.git)
