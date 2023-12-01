# Description
For of microsoft/unilm (https://github.com/microsoft/unilm/tree/master) with slight code modifications. For details on unilm and the content refer to the original repo.
The fork is taken from committ 24ea210.

# Modification
Here we list all the applied modifications.

## LayoutLMv3
- unilm/layoutlmv3/examples/object_detection/train_net.py
  
  It is now possible to log to wandb. Many options were added for setting the project name, run name, run id, and a custom wandb evaluation hook.

- layoutlmv3/examples/object_detection/ditod/mytrainer.py

  The custom event WandbCommonMetricLogger is defined and added to default_writers. Now the model logs to wandb the losses and the learning rate to wandb during training.

- layoutlmv3/examples/object_detection/cascade_layoutlmv3.yaml

  Configs are modified to accomodate the WANDB configurations.
