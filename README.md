# Auditory Slow-Fast

This repository implements the model proposed in the paper:

Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen, **Slow-Fast Auditory Streams for Audio Recognition**, *ICASSP*, 2021

[Project's webpage](https://ekazakos.github.io/auditoryslowfast/)

[arXiv paper](https://arxiv.org/abs/2103.03516)

## Citing

When using this code, kindly reference:

```
@ARTICLE{Kazakos2021SlowFastAuditory,
   title={Slow-Fast Auditory Streams For Audio Recognition},
   author={Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima},
           journal   = {CoRR},
           volume    = {abs/2103.03516},
           year      = {2021},
           ee        = {https://arxiv.org/abs/2103.03516},
}
```



## Pretrained models

You can download our pretrained models on VGG-Sound and EPIC-KITCHENS-100:
- Slow-Fast (EPIC-KITCHENS-100) [link](https://www.dropbox.com/s/cr0c6xdaggc2wzz/SLOWFAST_EPIC.pyth?dl=0)
- Slow (EPIC-KITCHENS-100) [link](https://www.dropbox.com/s/b1qaq8huu7heofp/SLOW_EPIC.pyth?dl=0)
- Fast (EPIC-KITCHENS-100) [link](https://www.dropbox.com/s/3qgwqsupqmsybai/FAST_EPIC.pyth?dl=0)
- Slow-Fast (VGG-Sound) [link](https://www.dropbox.com/s/oexan0vv01eqy0k/SLOWFAST_VGG.pyth?dl=0)
- Slow (VGG-Sound) [link](https://www.dropbox.com/s/4jcgozjenjwfo9k/SLOW_VGG.pyth?dl=0)
- Fast (VGG-Sound) [link](https://www.dropbox.com/s/vk123kwrphi7mer/FAST_VGG.pyth?dl=0)

## Preparation

* Requirements:
  * [PyTorch](https://pytorch.org) 1.7.1
  * [librosa](https://librosa.org): `conda install -c conda-forge librosa`
  * [h5py](https://www.h5py.org): `conda install h5py`
  * [wandb](https://wandb.ai/site): `pip install wandb`
  * [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
  * simplejson: `pip install simplejson`
  * psutil: `pip install psutil`
  * tensorboard: `pip install tensorboard` 
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/auditory-slow-fast/slowfast:$PYTHONPATH
```
* VGG-Sound:
  1. Download the audio. For instructions see [here](https://github.com/hche11/VGGSound)
  2. Download `train.pkl` ([link](https://www.dropbox.com/s/j60wkrcfdkfbvp9/train.pkl?dl=0)) and `test.pkl` ([link](https://www.dropbox.com/s/57rxp8wlgcqjbnd/test.pkl?dl=0)). I converted the original `train.csv` and `test.csv` (found [here](https://github.com/hche11/VGGSound/tree/master/data)) to pickle files with column names for easier use
* EPIC-KITCHENS:
  1. From the annotation repository of EPIC-KITCHENS-100 ([link](https://github.com/epic-kitchens/epic-kitchens-100-annotations)), download: `EPIC_100_train.pkl`, `EPIC_100_validation.pkl`, and `EPIC_100_test_timestamps.pkl`. `EPIC_100_train.pkl` and `EPIC_100_validation.pkl` will be used for training/validation, while `EPIC_100_test_timestamps.pkl` can be used to obtain the scores to submit in the AR challenge.
  2. Download all the videos of EPIC-KITCHENS-100 using the download scripts found [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts), where you can also find detailed instructions on using the scripts.
  3. Extract audio from the videos by running:
  ```
  python audio_extraction/extract_audio.py /path/to/videos /output/path 
  ```
  4. Save audio in HDF5 format by running:
  ```
  python audio_extraction/wav_to_hdf5.py /path/to/audio /output/hdf5/EPIC-KITCHENS-100_audio.hdf5
  ```

## Training/validation on EPIC-KITCHENS-100
To train the model run (fine-tuning from VGG-Sound pretrained model):
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.CHECKPOINT_FILE_PATH /path/to/VGG-Sound/pretrained/model
```
To train from scratch remove `TRAIN.CHECKPOINT_FILE_PATH /path/to/VGG-Sound/pretrained/model`.

You can also train the individual streams. For example, for training Slow run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOW_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.CHECKPOINT_FILE_PATH /path/to/VGG-Sound/pretrained/model
```

To validate the model run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

To obtain scores on the test set run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth 
EPICKITCHENS.TEST_LIST EPIC_100_test_timestamps.pkl EPICKITCHENS.TEST_SPLIT test
```

## Training/validation on VGG-Sound
To train the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations 
```

To validate the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).

