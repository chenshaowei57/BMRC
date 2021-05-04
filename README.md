# BMRC

Code and data of the paper "Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction, AAAI 2021" (https://arxiv.org/pdf/2103.07665.pdf)

Authors: 	Shaowei Chen, Yu Wang, Jie Liu, Yuelin Wang

#### Requirements:

```
  python==3.6.9
  torch==1.2.0
  transformers==2.9.0
```

#### Original Datasets:

You can download the 14-Res, 14-Lap, 15-Res, 16-Res datasets from https://github.com/xuuuluuu/SemEval-Triplet-data.

#### Data Preprocess:

```
  python dataProcess.py
  python makeData_dual.py
  python makeData_standard.py
```

#### How to run:

```
  python main.py --mode train # For training
```
