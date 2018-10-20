#!/usr/bin/env bash

python  split_train_valid.py

python  train_pytorch_128.py

python  train_pytorch_128_lovasz_loss.py

python  train_pytorch_128_lovasz_loss_clr.py

python  predict_pytorch_128.py

python  predict_pytorch_128_snapshot.py

