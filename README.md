<div align="center">

<h1>From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning</h1>

<p><strong>ICML 2026</strong></p>

<p>
  <a href="https://arxiv.org/abs/2604.10517"><img src="https://img.shields.io/badge/Paper-arXiv%202604.10517-b31b1b.svg" alt="Paper"></a>
  <a href="https://mp.weixin.qq.com/s/UcP8uGbUPiWJhrkQw_ffew"><img src="https://img.shields.io/badge/WeChat-%40%E6%9C%BA%E5%99%A8%E4%B9%8B%E5%BF%83-black%3Flogo%3Dwechat%26amp%3BlogoColor%3D07C160" alt="WeChat @机器之心"></a>
</p>

</div>

EgoTSR is a curriculum-based framework for ego-centric task-oriented spatiotemporal reasoning. It aims to help vision-language models move from explicit spatial understanding to internalized task-state judgment and finally to long-horizon planning, reducing chronological bias and spatiotemporal hallucinations in embodied tasks.

## Highlights

- A three-stage curriculum learning framework for egocentric task reasoning: CoT supervision, weakly supervised task-state tagging, and long-horizon planning.
- EgoTSR-Data, a 46M-sample dataset organized into CoT, Tag, and LongTag stages.
- A Reasoning-Enhanced Task Decomposition mechanism that maps high-level task descriptions into causal atomic subtask sequences.
- A dual-level evaluation framework covering both short atomic spatial perception and long-horizon logical planning.
- Strong performance on long-horizon logical reasoning and fine-grained perceptual reasoning across human demonstrations, simulations, and real-robot settings.

## Repository Structure

```text
EgoTSR/
├── 1get_clips.py              # Extract frames/clips from raw observations
├── 2group.py                  # Group clips by task name
├── 3refine.py                 # Refine task-specific clip endings
├── datamaker.py               # Optional CoT data generation helper
├── qwenvl/full/               # Qwen2.5-VL training and inference scripts
└── README.md
```

## Data Preparation

The original data processing pipeline follows three steps:

```bash
python 1get_clips.py
python 2group.py
python 3refine.py
```

The expected processed short-task structure is:

```text
EgoTSR-Short/
├── clips/
│   └── <scene_id>/<episode_id>/<task_name>/<frame_id>.png
├── observation/
│   └── <scene_id>/<episode_id>/
├── task/
│   └── *.jsonl
└── qwenvl/
    └── full/
```

Processed data links:

- Short-CoT and Short-Tag: [Hugging Face](https://huggingface.co/datasets/xuejingyang/SpatialLogic-Processed)
- Long-Tag: [Hugging Face](https://huggingface.co/datasets/EvanSirius/LOVE-Agibot-Beta)

Please update the hard-coded paths in the preprocessing scripts according to your local data layout before running them.

## Training

The Qwen2.5-VL training entrypoints are under `qwenvl/full/`.

CoT-stage training:

```bash
bash qwenvl/full/pdsh_cot_only.sh
```

Tag-stage training:

```bash
bash qwenvl/full/pdsh_tag_only.sh
```

Long-horizon Tag training:

```bash
bash qwenvl/full/pdsh_tag_long.sh
```

General-stage training:

```bash
bash qwenvl/full/pdsh_tag_general.sh
```

These scripts contain cluster-specific paths, node settings, model paths, and data paths. Please revise them for your own environment before use.

## Inference

The evaluation input CSV should contain fields such as `video_path1`, `video_path2`, and `task_name`. The output CSV stores the predicted target field, such as `img1` or `img2`.

CoT-model inference:

```bash
python qwenvl/full/inference4eval_8gpu_cot.py
```

Tag-model inference:

```bash
python qwenvl/full/inference4eval_8gpu_tag.py
```

Long-horizon model inference:

```bash
python qwenvl/full/inference4eval_8gpu_long.py
```

Please update checkpoint paths, base directories, input CSV paths, and output CSV paths in the scripts before running inference.

## Citation

If you find this project useful for your research, please cite:

```bibtex
@article{yang2026egotsr,
  title={From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning},
  author={Yang, Xiaoda and Liu, Yuxiang and Gao, Shenzhou and Wang, Can and Xue, Jingyang and Yang, Lixin and Mu, Yao and Jin, Tao and Zhang, Zhimeng and Yan, Shuicheng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2604.10517},
  year={2026}
}
```

## Acknowledgement

This project builds on Qwen2.5-VL and related open-source vision-language tooling. We thank the community for its contributions to multimodal reasoning, embodied AI, and long-horizon planning.
