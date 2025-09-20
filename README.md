<h1 align="center">
ComfyMind: Toward General-Purpose Generation 

via Tree-Based Planning and Reactive Feedback
</h1>
<p align="center">
    <a href="https://litaoguo.github.io/ComfyMind.github.io/"> <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-green"> </a>
    <a href="https://arxiv.org/abs/2505.17908"> <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-red"> </a>
    <a href="https://envision-research.hkust-gz.edu.cn/ComfyMind/"> <img alt="Demo" src="https://img.shields.io/badge/Demo-Live-orange"> </a>
</p>

## 💫 Introduction

![Teaser](assets/teaser.png)

We present ComfyMind, a collaborative AI system designed to enable robust and scalable general-purpose generation, built on the ComfyUI platform. We evaluate ComfyMind on three public benchmarks: ComfyBench, GenEval, and Reason-Edit, which span generation, editing, and reasoning tasks. Results show that ComfyMind consistently outperforms existing open-source baselines and achieves performance comparable to GPT-Image-1. ComfyMind paves a promising path for the development of open-source general-purpose generative AI systems.


## 📰 News

- **[2025/09/20]** We have released the evaluation code and the results of ComfyMind on evaluation benchmarks.
- **[2025/09/19]** ComfyMind has been accepted by NeurIPS 2025 🎉🎉🎉
- **[2025/05/30]** Our online demo has been released. [https://envision-research.hkust-gz.edu.cn/ComfyMind/](https://envision-research.hkust-gz.edu.cn/ComfyMind/)
- **[2025/05/24]** Our paper is submitted to arXiv.

## ️⚙️ Installation

### Step-by-step Installation

1. Clone the repository, create and activate conda environment:
```bash
git clone https://github.com/LitaoGuo/ComfyMind.git
cd ComfyMind

conda create -n comfymind python=3.12
conda activate comfymind
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

⚠️ 3. Before using ComfyMind, please prepare your ComfyUI with necessary models and extensions.

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) Installation
- [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) offers custom nodes management and installation.
- [Hugging Face](https://huggingface.co/) offers Models Installation 


## Configuration

⚠️ Modify `config.yaml` to set your APIs:


## 🚀 Usage

### Execute the main script
```bash
python main.py \
    --instruction "The generation instruction" \
    --resource1 "<optional>path/to/the/reference1" \
    --resource2 "<optional>path/to/the/reference2"
    --save_path "path/to/save/result"
```

### Gradio Demo
```bash
python main_gradio.py
```

### Evaluation
The results of ComfyMind on evaluation benchmarks can be viewed via [https://drive.google.com/drive/folders/1pR5vCQoo-W0Tr3vodyfQEWrI3om15Dak?usp=drive_link](https://drive.google.com/drive/folders/1pR5vCQoo-W0Tr3vodyfQEWrI3om15Dak?usp=drive_link).

To run the evaluation for ComfyMind, you can execute the following commands:
`python Evaluation/eval_geneval.py` , 
`python Evaluation/eval_reason_edit.py`

## 📜 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@article{guo2025comfymind,
  title={ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback},
  author={Guo, Litao and Xu, Xinli and Wang, Luozhou and Lin, Jiantao and Zhou, Jinsong and Zhang, Zixin and Su, Bolan and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2505.17908},
  year={2025}
}
```

## ❤️ Acknowledgments
We would like to thank the authors of the following projects for their excellent works.

- [ComfyBench](https://github.com/xxyQwQ/ComfyBench)
- [GPT-ImgEval](https://github.com/PicoTrex/GPT-ImgEval)

## 📄 License

This code is released under the MIT License.

## 📞 Contact

If you have any questions, please raise an issue or contact us at guolitauo@gmail.com.
