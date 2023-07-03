# Evaluating the adversarial robustness of human motion prediction

This is the code for the paper

Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin
[Evaluating the adversarial robustness of human motion prediction](https://arxiv.org/abs/2306.11990).

This code includes the adversarial attack method against convolutional models for human motion prediction and use the pytorch version of [TrajectoryCNN](https://github.com/lily2lab/TrajectoryCNN.git) as an example.

First things first, clone this repo and create required paths.

```bash
git clone https://github.com/ChengxuDuan/advHMP.git
cd advHMP
mkdir data
mkdir save_model
mkdir seq
```

## Requirements

* python 3.7
* CUDA 11.0
* Required packages are in the `requirements.txt`. You can install them by the following command:
```bash
cd [Path to the folder]/advHMP
pip install -r requirements.txt
```

## Prepare data
This code utilizes the data as the same as [_Learning Trajectory Dependencies for Human Motion Prediction_](https://arxiv.org/abs/1908.05436). In ICCV 19. You can find their code [here](https://github.com/wei-mao-2019/LearnTrajDep/tree/master).

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Then you need to move the data into `./data` like:
```
-data
 |-3DPW
   |-sequenceFiles
     |-test
     |-train
     |-validation
 |-CMU
   |-test
   |-train
 |-H36M
   |-S1
   |-S5
   |-S6
   |-S7
   |-S8
   |-S9
   |-S11
```
If you want to put the datasets elsewhere, please modify the setting in `args.py`

## Prepare models

You can download our **pretrained** model directly [here](https://drive.google.com/drive/folders/1zYdZdqOziPweEMfCg82tpJHAPHJgtwKc?usp=drive_link) and move them directly in `./save_model`.

Or, you can train from scratch by `train_pytorch.py`. You can run the commends below to train short-term TrajectoryCNN models on corresponding dataset:
* For Human 3.6m
```bash
python train_pytorch.py --dataset H36M --input_len 10 --seq_len 20 --joints_input 22 --train_batch 64
```
* For CMU Mocap
```bash
python train_pytorch.py --dataset CMU --input_len 10 --seq_len 20 --joints_input 25 --train_batch 16 
```
* For 3DPW
```bash
python train_pytorch.py --dataset 3DPW --input_len 10 --seq_len 25 --joints_input 24 --train_batch 16 
```
After you train the models, you can find them in `./save_model`. If you want to learn more parameters, you can find them in `args.py`.
This code support the function of checkpoint and you can enable it by the command `--is_load`. You also need to write the file name of the saved checkpoint at the line 130 of `train_pytorch.py`, then you can continue to train the model.

## Test the model cleanly and adversarially

After you get the models ready, you can test the model by `test_pytorch.py`. For example, you can run the following command to test a short-term TrajectoryCNN in both clean and adversarial situation on Human 3.6m:
```bash
python test_pytorch.py  --dataset H36M --input_len 10 --seq_len 20 --joints_input 22 --epsilon 1e-2 --iters 50 --epsilon_size 1e-3
```
Then you can find the results in the excel file `H36M_TrajCNN_Rec.csv`. It contains the errors of all the activities predicted by the target model. The average results may look like:

| interval(ms) | 80 | 160 | 320 | 400 |
|--|--|--|--|--|
| clean | 10.28249553 | 23.59175549 | 50.3486908 | 60.57837257 |
| adversarial | 40.30489464| 93.25951742 | 187.2042898 | 208.0358124 |

If you want to modify the path to your model, you can find the path at line 122 of `test_pytorch.py`.

If you want to pick which frames of the input motion sequence are perturbed, you can use the command `--attack_range 0 1 2`, which means you perturb the 1st, 2nd and 3rd frame of the input.

As some model may need some operation on the input motion sequence, you also need to apply the same operations in the optimization procedure of the attack. You can find more details and apply the operations in `Attack_3d.py`.

## Visualize the motion

As the attack results is not clear by the quantitative results, you can visualize the motion into images and animations by `draw_poses.py`.

You can run the commend below to generate the images:
```bash
python draw_poses.py --dataset H36M --activity walking
```
Then you can draw the images of the clean and perturbed motion.  Because 3DPW data we use are not divided in to different activities, drawing poses of 3DPW doesn't need `--activity` command.

The GIF may look like the examples below(solid lines are the input and predicted poses, while dotted lines are the ground truth):
![clean_example](https://github.com/ChengxuDuan/advHMP/blob/master/clean_example.gif)
![pert_example](https://github.com/ChengxuDuan/advHMP/blob/master/pert_example.gif)
## Citing

If you use our code, please cite our work.

```
@misc{duan2023evaluating,
      title={Evaluating Adversarial Robustness of Convolution-based Human Motion Prediction}, 
      author={Chengxu Duan and Zhicheng Zhang and Xiaoli Liu and Yonghao Dang and Jianqin Yin},
      year={2023},
      eprint={2306.11990},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

Some of our evaluation code and data process code was adapted/ported from [LTD](https://github.com/wei-mao-2019/LearnTrajDep/tree/master)

