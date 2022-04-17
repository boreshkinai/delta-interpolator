[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motion-inbetweening-via-deep-d-interpolator/motion-synthesis-on-lafan1)](https://paperswithcode.com/sota/motion-synthesis-on-lafan1?p=motion-inbetweening-via-deep-d-interpolator)


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



## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone https://github.com/boreshkinai/delta-interpolator```

## Build docker image and launch container

Build image and start the lightweight docker container. Note that this assumes that the data for the project will be stored in the shared folder /home/pose-estimation accessible to you and other project members. 
```
docker build -f Dockerfile -t delta_interpolator:$USER .

nvidia-docker run -p 18888:8888 -p 16006:6006 -v ~/workspace/delta-interpolator:/workspace/delta-interpolator -t -d --shm-size="8g" --name delta_interpolator_$USER delta_interpolator:$USER
```

## Enter docker container and launch training session

```
docker exec -i -t delta_interpolator_$USER  /bin/bash 
```
Once inside docker container, this launches the training session for the proposed model. Checkpoints and tensorboard logs are stored in ./logs/lafan/transformer
```
python run.py --config=src/configs/transformer.yaml
```
This evaluates zero-velocity and the interpolator models
```
python run.py --config=src/configs/interpolator.yaml
python run.py --config=src/configs/zerovel.yaml
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
```
https://storage.googleapis.com/delta-interpolator/pretrained_model.zip
```
