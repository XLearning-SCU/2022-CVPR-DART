
PyTorch implementation for [Learning with Twin Noisy Labels for Visible-Infrared Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Learning_With_Twin_Noisy_Labels_for_Visible-Infrared_Person_Re-Identification_CVPR_2022_paper.pdf) (CVPR 2022).


## Introduction

### DART framework
<img src="https://github.com/XLearning-SCU/2022-CVPR-DART/blob/main/figs/framework.png"  width="760" height="268" />

## Requirements

- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
## Datasets

### SYSU-MM01 and RegDB
We follow [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) to obtain datasets.

## Training and Evaluation

### Training

Modify the ```data_path``` and  specify the ```noise_ratio``` to train the model.

```train
# SYSU-MM01: noise_ratio = {0, 0.2, 0.5}
python run.py --gpu 0 --dataset sysu --data-path data_path --noise-rate 0.2 --savename sysu_dart_nr20 

# RegDB: noise_ratio = {0, 0.2, 0.5}, trial = 1-10
python run.py --gpu 0 --dataset regdb --data-path data_path --noise-rate 0.2 --savename regdb_dart_nr20 --trial 1
```
### Evaluation

Modify the  ```data_path``` and ```model_path``` to evaluate the trained model. 

```
# SYSU-MM01: mode = {all, indoor}
python test.py --gpu 0 --dataset sysu --data-path data-path --model_path model_path --resume-net1 'sysu_dart_nr20_net1.t' --resume-net2 'sysu_dart_nr20_net2.t' --mode all

# RegDB: --tvsearch or not (whether thermal to visible search)
python test.py --gpu 0 --dataset regdb --data-path data-path --model_path model_path --resume-net1 'regdb_dart_nr20_trial{}_net1.t' --resume-net2 'regdb_dart_nr20_trial{}_net2.t'
```


## Citation

If DART is useful for your research, please cite the following paper:
```
@InProceedings{Yang_2022_CVPR,
    author={Yang, Mouxing and Huang, Zhenyu and Hu, Peng and Li, Taihao and Lv, Jiancheng and Peng, Xi},
    title={Learning With Twin Noisy Labels for Visible-Infrared Person Re-Identification},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={14308-14317}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) licensed under Apache 2.0.
