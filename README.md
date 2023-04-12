# Learning Human Motion with Deep Delta Interpolatation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motion-inbetweening-via-deep-d-interpolator/motion-synthesis-on-lafan1)](https://paperswithcode.com/sota/motion-synthesis-on-lafan1?p=motion-inbetweening-via-deep-d-interpolator)

<p align="center">
  <img width="1200"  src=./fig/Fig1_Comp3_PushAndStumble_start5381.png>
  <figcaption align = "center"><b>Fig.1 - Comparison of human motion prediction. Left: Our model. Center: Linear interpolation.     Right: Ground Truth. Tracer lines represent evolution of joint locations over time.</b></figcaption>
</p>

<p align="center">
  <img width="400"  src=./fig/anim_1.gif \> 
  <img width="400"  src=./fig/anim_2.gif \>
</p>
<figcaption align = "center"><b>Fig.2 - Left: The demonstration of the proposed $\Delta$-interpolator approach (green), Linear interpolator baseline (yellow), ground truth motion (white). Positional errors are indicated in red. Right: Robustness of the $\Delta$-interpolator (green) w.r.t. the out-of-distribution operation. Ground truth motion (white), SLERP interpolator (yellow), proposed model with both input and output delta modes disabled (blue). $\Delta$-interpolator is not affected by distribution shift. </b></figcaption>

<br/><br/>

If you use this code in any context, please cite the following paper:
```
@misc{oreshkin2022motion,
      title={Motion Inbetweening via Deep $\Delta$-Interpolator}, 
      author={Boris N. Oreshkin and Antonios Valkanas and Félix G. Harvey and Louis-Simon Ménard and Florent Bocquelet and Mark J. Coates},
      year={2022},
      eprint={2201.06701},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Using this repository

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone https://github.com/boreshkinai/delta-interpolator```

## Build docker image and launch container 
```
docker build -f Dockerfile -t delta_interpolator:$USER .
nvidia-docker run -p 8888:8888 -p 6006:6006 -v ~/workspace/delta-interpolator:/workspace/delta-interpolator -t -d --shm-size="1g" --name delta_interpolator_$USER delta_interpolator:$USER
```

## Enter docker container and launch training session

```
docker exec -i -t delta_interpolator_$USER  /bin/bash 
```
Once inside docker container, this launches the training session for the proposed model. Checkpoints and tensorboard logs are stored in ./logs/lafan/transformer
```
python run.py --config=src/configs/transformer.yaml
```
This evaluates zero-velocity and the interpolator models for LaFAN1
```
python run.py --config=src/configs/interpolator.yaml
python run.py --config=src/configs/zerovel.yaml
```
To run the Anidance benchmark experiments run:
```
python run.py --config=src/configs/transformer_infill.yaml
```
For zero-velocity and interpolator baselines run:
```
python run.py --config=src/configs/interpolator_anidance.yaml
python run.py --config=src/configs/zerovel_anidance.yaml
```

Training losses eveolve as follows:
<p align="center">
  <img width="1200"  src=./fig/train_losses.png>
</p>

## Open the results notebook to look at the metrics:
```
http://your_server_ip:18888/notebooks/LaFAN1Results.ipynb
```
The notebook password is `default`

## Pretrained model is available here

[https://storage.googleapis.com/delta-interpolator/pretrained_model.zip](https://storage.googleapis.com/delta-interpolator/pretrained_model.zip)


