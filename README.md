# [APNTracking]

## 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test
Download pretrained model and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```
python test.py 	                          \
	--trackername SiamAPN           \ # tracker_name
	--dataset UAV10fps                  \ # dataset_name
	--snapshot snapshot/general_model.pth   # model_path
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 3. Train

### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://pan.baidu.com/s/1ZTdfqvhIRneGFXur-sCjgg) (code: t7j8)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


### Train a model
To train the SiamAPN model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

## 4. Trackers

## [SiamAPN] 
The pre-trained model can be found at (epoch=37) : [general_model](https://pan.baidu.com/s/1GSgj3UwObcUKyT8TFSJ5qA)(code:w3u5) 

We provide the tracking [results_v1](https://pan.baidu.com/s/1EWOSHNcOldJBCCmwY-mvVA) (code: s3p1) of UAV123@10fps, UAV20L, and VisDrone2018-SOT-test. Besides, the tracking [results_v2](https://pan.baidu.com/s/1zCmiWHbNiDTyUELyZ8NXwg) (code: j4t5) of UAV123@10fps, UAV20L, VisDrone2018-SOT-test and UAVTrack112 are  also provided. 

## [ADSiamAPN] 
The pre-trained model can be found at (epoch=25): [general_model](https://pan.baidu.com/s/1ovv45-pfQ9PQQJMi2b2K3A)(code:j29k)

We provide the tracking [results](https://pan.baidu.com/s/11Gpf4vjKrIyWh4QV8CVWTA) (code: xb41) of UAV123@10fps, UAV20L.

## 5. Evaluation 
If you want to evaluate the tracker mentioned above, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV10fps                  \ # dataset_name
	--tracker_prefix 'general_model'   # tracker_name
```
## 6. UAVTrack112 benchmark
UAVTrack112 benchmark is created from images captured during the real-world tests. It can be downloaded at [UAVTrack112](https://pan.baidu.com/s/1lF2pTQu39dIUC7iGR44mxA) (code: jk29).

## 7. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.