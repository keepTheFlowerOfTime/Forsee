# Forsee
The artifact of ISSTA2024 Paper: Enhancing Robustness of Code Authorship Attribution through Expert Feature Knowledge 

first update: 2024-09-25
second update: 2024-11-08

### Step1:

clone the tree-sitter module to /libs
```
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
```

### Step2:

install all the requirements for the Repo

```
pip install -r requirements.txt
```

### Step3:

unzip the dataset in ./dataset, for example:
```
unzip gcj_cpp.zip
```

### Step4:

run test file for test/tree_sitter_test.py to validate tree-sitter is enable

```
python test/tree_sitter_test.py
```

### Step5:

run cmd_batch.py for test

```
python cmd_batch.py
```

for epoch_num=150, the top1 of Forsee is 0.823 (just for test)

the Training Step contains 4 models:
1. Layout Indepentent Model
2. Lexical Indepentent Model
3. Syntact Indepentent Model
4. Forsee Framework


## Reference
```
@inproceedings{10.1145/3650212.3652121,
author = {Guo, Xiaowei and Fu, Cai and Chen, Juan and Liu, Hongle and Han, Lansheng and Li, Wenjin},
title = {Enhancing Robustness of Code Authorship Attribution through Expert Feature Knowledge},
year = {2024},
isbn = {9798400706127},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3650212.3652121},
doi = {10.1145/3650212.3652121},
pages = {199–209},
numpages = {11},
keywords = {code authorship attribution, expert knowledge, machine learning, robustness},
location = {Vienna, Austria},
series = {ISSTA 2024}
}
```