# Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast

This is a PyTorch implementation for CMPC, as described in our paper:


**[Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast](https://arxiv.org/pdf/2004.12943.pdf)**  
```angular2html
@inproceedings{PCL,
<!---->
}
```

![Framework](./img/fig_pipeline.png)


We also provide the [pretrained model](#unsupervised-training) and [testing resources](#testing-data).
### Unsupervised Training



### Requirments:

* torch==1.7.0+cu110
* matplotlib==3.4.3
* pykeops==1.5
* pandas==1.1.3
* numpy==1.21.4
* librosa==0.6.2
* ipython==8.0.1
* Pillow==9.0.1
* PyYAML==6.0
* scikit_learn==1.0.2

### Download Pre-trained Models
<a href="https://drive.google.com/file/d/1pBfEaKVrdNNCcyV7_sXcCmuXgW0j_b94/view?usp=sharing">CID</a>| <a href="https://drive.google.com/file/d/1Bm065HClTvo6T4mjlWP1b0AC6UdRzFHh/view?usp=sharing">CMPC</a>
------ | ------


### Unsupervised Training
```angular2html
cd experiments/cmpc
python train.py CONFIG.yaml
```

### Evalution on our trained model
```angular2html
cd experiments/cmpc
python matching.py CONFIG.yaml --ckp_path 'checkpoint path'
python verfication.py CONFIG.yaml --ckp_path 'checkpoint path'
python retrieval.py CONFIG.yaml --ckp_path 'checkpoint path'
```

### Testing data

[Matching](./data/matching), [verification](./data/veriflist) and [retrieval](./data/retrieval) testing data is released at [./data](./data) directory.