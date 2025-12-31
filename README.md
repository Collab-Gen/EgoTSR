
# 原始数据处理流程（憬阳整理下）
## SpatialLogic-Short
1. 下载数据集：https://huggingface.co/datasets/yangxiaoda/SpacialLogic
2. 解压observation中的文件，删掉原压缩文件
3. 把observation中的视频以10倍下采样切分成帧，制作clips文件夹：python 1get_clips.py
4. 把clips文件夹的每个文件根据任务名称归类：python 2group.py
5. 由于“Place”指令对应的视频结尾部分多余，需要减除：python 3refine.py
6. 拉取项目：
git clone https://github.com/Yangxiaoda1/SpatialLogic.git
7. 此时，文件结构为：
```bash
|-SpacialLogic-Short
    |-clips
        |-327
            |-648642
                |-Place the held corn into the shopping cart's plastic bag.
                    |-1.png
                    |-2.png
    |-observation
        |-327
            |-648642
        |-...
    |-task
        |-.jsonl
    |-qwenvl
        |-full
            |-mycheckpoint
            |-train.py
            |-inference.py
```

[Optional] Cot数据生成：调用GPT接管，然后人工矫正
```bash
python datamaker.py
```


## SpatialLogic-Long
1. 

# 处理好的数据
Short-CoT,Short-Tag: https://huggingface.co/datasets/xuejingyang/SpatialLogic-Processed

Long-Tag: https://huggingface.co/datasets/EvanSirius/LOVE-Agibot-Beta

# 训练
## SpatialLogic-CoT
```bash
bash /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/pdsh_cot_only.sh
```

## SpatialLogic-Tag
```bash
bash /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/pdsh_tag_only.sh
```

## SpatialLogic-Long
```bash
bash /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/pdsh_tag_long.sh
```

## SpatialLogic-General (Future Work)
```bash
bash /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/pdsh_tag_general.sh
```

## SpatialLogic-Equal (Can Wang)


# 推理
原材料：gt.csv，里面包含video_path1, video_path2, task_name几个字段

输出：pred.csv，target字段 [img1/img2]

## SpatialLogic-CoT

```bash
python /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/inference4eval_8gpu_cot.py
```

## SpatialLogic-Tag
```bash
python /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/inference4eval_8gpu_tag.py
```

## SpatialLogic-Long
```bash
python /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/SpatialLogic/qwenvl/full/inference4eval_8gpu_long.py
```

# 算子
## SpatialLogic-General Filter (Menglan Tang && Jingyang Xue)

## SaptialLogic-Subtask Guidance (Shenzhou Gao)

# 引用
