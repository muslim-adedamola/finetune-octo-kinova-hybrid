# Finetune Octo on Kinova Gen 3 (7 DoF) robot with Hybrid Arm-Gripper Control

## What is Octo?

[Octo](https://octo-models.github.io/) is an open-source generalist robot policy for manipulation, sometimes described as a robot foundation model or vision-language-action policy. It is pretrained on large-scale robot demonstration data ([OpenX Embodiment](https://robotics-transformer-x.github.io/)) and is designed to support flexible task definitions, observations, and action spaces. Octo can be conditioned on language instructions or goal images and can be finetuned to new robot setups with relatively small in-domain datasets.

In this repository, **Octo** is used in goal-image-conditioned mode and finetuned on a Kinova Gen3 (7 DoF) manipulator demonstrations for bottle pick-and-lift.

This repository contains a practical research-engineering pipeline for adapting Octo to a Kinova robot using a hybrid action design:

- **Octo diffusion action head** for continuous arm motion
- **Binary BCE gripper head** for open/close gripper prediction
- **Goal-image conditioning**
- **TFDS/RLDS dataset conversion**
- **Offline per-episode evaluation**
- **Real-robot deployment with latency-aware automatic receding-horizon control**

> Status: research prototype validated on a Kinova robot.

## Demo

**demo1.mp4**

> Real-robot rollout shown at 2× speed.

---

## Pretrained Checkpoint

Downloadable checkpoint as used in demo videos is made available here:
> **Best checkpoint:** [Download from Google Drive](https://drive.google.com/file/d/1Eiv2iN5whw7qQazFhIcdwFJOwBmYghiL/view?usp=sharing)

Expected checkpoint structure after extraction:

```
best/
  checkpoint_*
```

Example Usage:

```
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path /path/to/downloaded/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3
  --execute_actions #to run on real robot. Remove to run in shadow mode
```

---

## Overview

The project finetunes a pretrained Octo model on a Kinova Gen 3 (7 DoF) demonstrations for a bottle pick-and-lift task. The training data was collected using a white bottle. Additional real-robot rollouts were tested on different white-bottle positions and unseen bottle appearances, including pink and black bottles.

### Camera Setup

An external Intel RealSense RGB camera observing the workspace from a third-person scene view.

Only the scene camera RGB image is used as the policy observation. No wrist camera and no proprioceptive robot-state observations are used as model inputs in the released finetuning/deployment pipeline.

The RealSense frame is cropped and resized to 256×256 so that the live inference input matches the cropped visual distribution used during training. The current implementation is Kinova-specific at the deployment layer because it uses the Kinova Kortex API. However, the dataset, training, and hybrid action-head design can be adapted to other manipulators with robot-specific data collection and deployment wrappers.

---

## Real-Robot Demos

| Demo | Description |
| --- | --- |
| [White bottle rollout](assets/demos/demo1.mp4) | In-distribution bottle used during training |
| [White bottle, new position](assets/demos/demo2.mp4) | Same bottle type, different initial position |
| [Additional white-bottle rollout](assets/demos/demo3.mp4) | Additional in-distribution rollout |
| [Pink bottle generalization](assets/demos/demo4.mp4) | Unseen bottle color |
| [Pink bottle, new position](assets/demos/demo5.mp4) | Unseen bottle color and different position |
| [Black bottle generalization](assets/demos/demo6.mp4) | Unseen bottle color and different position |

The policy was trained only on the white bottle, so unseen bottle-color rollouts should be interpreted as qualitative generalization tests.

---

## Method Summary

The pipeline follows this flow:

```
Raw Kinova Gen 3 demonstrations
→ TFDS/RLDS dataset builder
→ Octo-compatible standardization
→ Hybrid finetuning
→ Offline evaluation
→ Real-robot deployment
```

During training:

- the first 6 action dimensions are modeled by Octo's diffusion action head;
- the gripper action is modeled separately using a binary BCE head;
- goal images are passed as task conditioning;
- the model is trained on windowed image observations and action chunks.

During deployment:

- RealSense provides RGB observations;
- images are cropped/resized to 256×256;
- the goal image is loaded once and used as task conditioning;
- arm motion uses selected action chunks from Octo;
- gripper state uses the selected BCE gripper chunk;
- the controller runs in `auto_receding_horizon` mode.

Rollout settings used:

```
--control_mode auto_receding_horizon \
--arm_chunk_index 1 \
--gripper_chunk_index 3
```

---

## Repository Structure

```
finetune-octo-kinova-hybrid/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── calibration/
│   │   ├── collect_calibration_data.py      ← stage 1: move arm + collect ChArUco frames
│   │   └── run_calibration.py               ← stage 2: solve for T_BASE_CAMERA
│   │
│   ├── collection/
│   │   ├── collect_episode.py               ← main data collection entry point
│   │   ├── episode_logger.py                ← EpisodeLogger, RealSenseLatestRGB
│   │   ├── trajectory_primitives.py         ← PD controller, motion primitives
│   │   ├── camera_config.py                 ← intrinsics, extrinsics, object constants
│   │   ├── detect_bottle_live.py            ← live detection diagnostic
│   │   └── arm_frame_grabber.py             ← threaded RTSP frame reader
│   │
│   ├── dataset/
│   │   ├── kinova_dataset/
│   │   │   ├── __init__.py
│   │   │   ├── kinova_dataset.py
│   │   │   └── kinova_dataset_dataset_builder.py
│   │   ├── kinova_standardize_octo.py
│   │   ├── print_episode_dir.py
│   │   └── validate_one_episode.py
│   │
│   ├── training/
│   │   └── finetune_hybrid_octo_bce.py
│   │
│   ├── evaluation/
│   │   ├── eval_hybrid_octo_bce.py
│   │   └── eval_gripper_traces.py
│   │
│   └── deployment/
│       ├── run_finetuned_hybrid_model.py
│       └── utilities.py
│
├── configs/
├── assets/
│   ├── demos/
│   └── images/
│
├── docs/
│   ├── COLLECTION.md
│   ├── CALIBRATION.md
│   ├── SETUP.md
│   ├── DATASET.md
│   ├── TRAINING.md
│   └── DEPLOYMENT.md
│
├── checkpoints/
└── data/
```

---

## Installation

This project assumes you already have a working Octo environment.

Clone this repository:

```
git clone https://github.com/muslim-adedamola/finetune-octo-kinova-hybrid.git
cd finetune-octo-kinova-hybrid
```

Install additional dependencies as needed:

```
pip install -r requirements.txt
```

Main dependencies include:

- Octo
- JAX / Flax / Optax
- TensorFlow and TensorFlow Datasets
- OpenCV
- NumPy / Pandas / Matplotlib
- Weights & Biases
- Intel RealSense SDK / Pyrealsense2
- Kinova Kortex API

The Kinova deployment script requires a working Kortex API installation and access to the robot over the network.

---

## Calibration

Before collecting episodes, the camera must be calibrated to obtain `T_BASE_CAMERA` — the transform from camera frame to robot base frame used for world-coordinate projection.

See **[docs/CALIBRATION.md](docs/CALIBRATION.md)** for the full two-stage hand-eye calibration procedure.

Quick start:

```
# Stage 1: collect paired (image, robot pose) data
python scripts/calibration/collect_calibration_data.py --n_poses 30

# Stage 2: solve for T_BASE_CAMERA
python scripts/calibration/run_calibration.py
```

Copy the printed matrix into `scripts/collection/camera_config.py`.

---

## Data Collection

The full scripted data collection pipeline is in `scripts/collection/`. It covers YOLOE-based bottle detection, world-coordinate projection, PD-controlled pick-and-lift execution, and per-timestep logging to the episode format used by the TFDS builder.

See **[docs/COLLECTION.md](docs/COLLECTION.md)** for setup instructions, calibration steps, and usage.

Quick start (one episode):

```
python scripts/collection/collect_episode.py \
  --out_root /path/to/kinova_bottle_lift_raw \
  --hz 10 \
  --lift_height 0.20 \
  --hold_time 2.0
```

Press **ESC** at any point to abort and discard the episode.

---

## Dataset Preparation

The raw Kinova Gen 3 (7 DoF) demonstrations are expected in this format:

```
<data_dir>/downloads/manual/kinova_dataset_2/episodes/
  episode_000/
    episode.csv
    rgb/
      000000.png
      000001.png
      ...
  episode_001/
    episode.csv
    rgb/
      000000.png
      000001.png
      ...
```

Build the TFDS/RLDS dataset:

```
tfds build scripts/dataset/kinova_dataset \
  --overwrite \
  --data_dir /path/to/tensorflow_datasets \
  --beam_pipeline_options="direct_running_mode=multi_threading,direct_num_workers=10"
```

Validate one episode:

```
python scripts/dataset/validate_one_episode.py \
  --data_dir /path/to/tensorflow_datasets \
  --config default \
  --split train
```

Print train/val episode split:

```
python scripts/dataset/print_episode_dir.py
```

---

## Dataset Availability

A converted TFDS/RLDS version of the Kinova bottle pick-and-lift dataset is available here (size is 936 MB):
> **Converted TFDS/RLDS dataset:** [Download from Google Drive](https://drive.google.com/file/d/1gVG1LpR4IPke7KlWEkE2oC9X1H8BMIbQ/view?usp=sharing)

Expected extracted structure:

```
kinova_dataset/
└── default/
    └── 0.1.5/
```

Just go ahead and paste this kinova_dataset/ folder into your tensorflow directory

Example Usage:

```
python scripts/training/finetune_hybrid_octo_bce.py \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --dataset_name kinova_dataset \
  --dataset_config default \
  --dataset_version 0.1.5
```

If you prefer to build the tfds/rlds version of the raw dataset demos (bottle pick and lift) yourself, The download link is here:
> **[Link to download Raw Dataset](https://drive.google.com/file/d/1DxAAcv0oV7qoBO6aO62Omon7kLr4Cql9/view?usp=sharing)**

To collect similar demos on your robot, see [docs/COLLECTION.md](docs/COLLECTION.md) for the full data collection pipeline, and [docs/DATASET.md](docs/DATASET.md) for the expected raw data format.

---

## Octo Standardization Function

The training and evaluation scripts expect the Kinova standardization function to be importable from the Octo package path:

```
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
```

Copy the standardization file into your local Octo source tree:

```
cp scripts/dataset/kinova_standardize_octo.py /path/to/octo/octo/data/kinova_standardize_octo.py
```

---

## Training

Run hybrid finetuning:

```
python scripts/training/finetune_hybrid_octo_bce.py \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --save_dir ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce \
  --dataset_name kinova_dataset \
  --dataset_config default \
  --dataset_version 0.1.5 \
  --batch_size 64 \
  --window_size 2 \
  --action_horizon 4 \
  --steps 50000 \
  --learning_rate 3e-5 \
  --gripper_bce_weight 5.0
```

The script saves:

```
checkpoints/
  finetune_ckpts_hybrid_arm_diffusion_gripper_bce/
    checkpoint_*
    hybrid_checkpoint_meta.json
    best/
      checkpoint_*
```

---

## Offline Evaluation

Run per-episode offline evaluation:

```
python scripts/evaluation/eval_hybrid_octo_bce.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --split val \
  --save_dir ./offline_eval_results_hybrid
```

Run gripper trace diagnostics:

```
python scripts/evaluation/eval_gripper_traces.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --split val \
  --save_dir ./offline_eval_gripper_traces \
  --save_gripper_traces \
  --num_trace_episodes 5
```

---

## Real-Robot Deployment

Run live inference in shadow mode first:

```
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3
```

To execute robot actions, add:

```
--execute_actions
```

Example execution command:

```
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --execute_actions \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3
```

Use `--crop_json_path` and `--goal_crop_json_path` to save/reuse deployment crops.

### Deployment Cropping

During data collection, the RealSense camera captured a wide view of the workspace, including other robots and background objects. The training episodes were cropped to focus on the active Kinova arm, tabletop workspace, and target object. For this reason, the live deployment script includes `--startup_crop_review`, `--crop_json_path`, and `--goal_crop_json_path` so the inference-time camera input can be aligned with the visual distribution used during training.

See [docs/DATASET.md](docs/DATASET.md) for details on the cropping workflow.

---

## Safety Notes

Real-robot deployment can be unsafe if the robot, workspace, camera crop, action scaling, or gripper logic is misconfigured.

Before using `--execute_actions`:

- verify the robot emergency stop is accessible;
- run in shadow mode first;
- confirm the model input crop;
- use conservative speed limits;
- keep the workspace clear;
- supervise the robot continuously.

The same precautions apply during data collection. See [docs/COLLECTION.md](docs/COLLECTION.md) for collection-specific safety guidance.

---

## Scope and Current Status

This repository is a research-engineering prototype for adapting Octo to a Kinova manipulator.

Current scope:

- The current release focuses on a single real-robot task: goal-image-conditioned bottle pick-and-lift.
- The policy was finetuned on approximately 110 task-specific Kinova Gen 3 (7 DoF) demonstrations, consistent with the data-efficient finetuning setting explored by Octo.
- Training demonstrations used a white bottle; additional rollouts on pink and black bottles are included as qualitative appearance-generalization tests.
- Unseen bottle-color rollouts are successful qualitative demonstrations; a larger quantitative generalization benchmark is left for future work.
- Real-robot deployment currently assumes a Kinova arm, Kinova Kortex API, and Intel RealSense RGB input.
- Multi-robot support would mainly require robot-specific data collection and deployment wrappers; the current release provides the Kinova implementation.
- The current deployed policy can exhibit some delay after reaching the bottle before closing the gripper. This behavior is visible in some rollouts and remains an open deployment-timing improvement.

## Future Work

Planned next steps include:

- improving deployment timing, especially the short delay observed after the robot reaches the bottle before closing the gripper;
- finetuning Octo on a longer-horizon Kinova manipulation task beyond single bottle pick-and-lift;
- extending the deployment layer to support additional robot arms through robot-specific wrappers.

---

## Acknowledgements

This project builds on Octo, an open-source generalist robot policy from the Octo Model Team. Octo provides pretrained checkpoints and finetuning infrastructure for adapting robot policies to new domains.

The Kinova TFDS/RLDS dataset builder in this repository is adapted from Karl Pertsch's [RLDS Dataset Builder](https://github.com/kpertsch/rlds_dataset_builder/tree/main), an example workflow for converting custom robot datasets into RLDS format for X-embodiment-style training.

---

## Citation

This repository builds on Octo. If you use this repository in academic work, please cite the original Octo paper and any relevant dataset/tooling sources.

[Link to Octo's Citation](https://github.com/octo-models/octo#citation)
