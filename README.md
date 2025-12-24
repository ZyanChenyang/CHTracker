# CHTracker

## Abstract

Multi-object tracking (MOT) has garnered considerable attention due to its relevance in practical applications such as automated devices in smart cities. However, under complex conditions, existing trackers often fail to accurately capture or characterize target motion patterns, exhibiting limitations in flexibility and interpretability. To address these challenges, this paper introduces CHTracker, a confidence-guided hierarchical association paradigm for MOT. By integrating spatial features with varying confidence levels, CHTracker enhances the granularity of motion pattern modeling in edge-case scenarios where conventional trackers are prone to association ambiguity. Our paradigm adaptively utilizes distinct tracking cues and assignment metrics tailored to hierarchical target structures, thereby enabling collaborative tracking. Additionally, CHTracker incorporates the diagonal length of the target bounding box as a state variable during position prediction, which significantly improves the robustness against diverse motion noise. Extensive experimental results on multiple benchmarks, including DanceTrack, MOT17, MOT20, and Singapore Maritime Dataset (SMD), demonstrate that CHTracker achieves the state-of-the-art performance in accuracy, robustness, and generalization. Furthermore, our association paradigm is extended to a visible-infrared fusion version for evaluation on the multimodal CAMEL dataset, underscoring its practical potential to fulfill heterogeneous modality requirements in real-world scenarios. 

### Highlights

- Maintains **Simple, Online and Real-Time (SORT)** characteristics.
- **Training-free** and **plug-and-play** manner.
- **Strong generalization** for diverse trackers and scenarios.

## Installation

CHTracker code is based on [OC-SORT](https://github.com/noahcao/OC_SORT), [HybridSORT](https://github.com/ymzis69/HybridSORT), and [FastReID](https://github.com/JDAI-CV/fast-reid). The ReID component is optional and based on [FastReID](https://github.com/JDAI-CV/fast-reid). Tested the code with Python 3.8 + Pytorch 1.10.0 + torchvision 0.11.0.

Step1. Install CHTracker

```shell
git clone https://github.com/ZyanChenyang/CHTracker.git
cd CHTracker
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```shell
pip3 install cython_bbox pandas xmltodict
```

Step4. [optional] FastReID Installation

You can refer to [FastReID Installation](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

```shell
pip install -r fast_reid/docs/requirements.txt
```

## Data preparation

**Our data structure is the same as [OC-SORT](https://github.com/noahcao/OC_SORT).** 

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [SMD](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset), [DanceTrack](https://github.com/DanceTrack/DanceTrack) and put them under <CHTracker_HOME>/datasets in the following structure (you can download YOLOX weights from [ByteTrack](https://github.com/ifzhang/ByteTrack) or [OC-SORT](https://github.com/noahcao/OC_SORT)) :

   ```
   datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————dancetrack        
            └——————train
               └——————train_seqmap.txt
            └——————val
               └——————val_seqmap.txt
            └——————test
               └——————test_seqmap.txt
   ```

2. Prepare DanceTrack dataset:

   ```python
   # replace "dance" with ethz/mot17/mot20/crowdhuman/cityperson/cuhk for others
   python3 tools/convert_dance_to_coco.py 
   ```

3. Prepare MOT17/MOT20 dataset. 

   ```python
   # build mixed training sets for MOT17 and MOT20 
   python3 tools/mix_data_{ablation/mot17/mot20}.py
   ```

4. [optional] Prepare ReID datasets:

   ```
   cd <CHTracker_HOME>
   
   # For DanceTrack
   python3 fast_reid/datasets/generate_cuhksysu_dance_patches.py --data_path <dataets_dir> 
   ```

## Model Zoo

Download and store the trained models in 'pretrained' folder as follow:

```
<CHTracker_HOME>/pretrained
```

### Detection Model

We provide some pretrained YOLO-X weights for CHTracker, which are inherited from [ByteTrack](https://github.com/ifzhang/ByteTrack).

| Dataset         | Model                                                        |
| --------------- | ------------------------------------------------------------ |
| DanceTrack-val  | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| DanceTrack-test | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-half-val  | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-test      | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT20-test      | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |


* For more YOLO-X weights, please refer to the model zoo of [ByteTrack](https://github.com/ifzhang/ByteTrack).

### ReID Model

Ours ReID models for **MOT17/MOT20** is the same as [BoT-SORT](https://github.com/NirAharon/BOT-SORT) , you can download from [MOT17-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), ReID models for DanceTrack is trained by ourself, you can download from [DanceTrack](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing).

**Notes**:

* [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) is trained by [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), because the weight from BOT-SORT is corrupted. Refer to [Issue](https://github.com/GerardMaggiolino/Deep-OC-SORT/issues/6).

## Tracking

### DanceTrack

**dancetrack-val dataset**


python tools/run_chtracker_dance.py -f exps/example/mot/yolox_dancetrack_val_chtracker.py -b 1 -d 1 --fuse

```

**dancetrack-test dataset**

```

python tools/run_chtracker_dance.py --test -f exps/example/mot/yolox_dancetrack_test_chtracker.py -b 1 -d 1 --fuse 

```

CHTracker is designed for online tracking, but offline interpolation has been demonstrated efficient for many cases and used by other online trackers:

```shell
# offline post-processing
python3 tools/interpolation.py $result_path $save_path
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BOT-SORT), [HybridSORT](https://github.com/ymzis69/HybridSORT), and [FastReID](https://github.com/JDAI-CV/fast-reid). Many thanks for their wonderful works.