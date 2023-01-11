"""
File: lgbm_optuna.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: Downloads
Last Version: <<projectversion>>
Relative Path: \lgbm_optuna.py
File: lgbm_optuna.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 8th November 2022 8:20:34 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# # LightGBM and Tunning parameters with Optuna Bayesian optimizationlgbm_mse
# Author: Man Chong Chan

import argparse
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
import math
from sklearn import metrics
import sys
import logging
from scipy import sparse
import random
from sklearn import metrics
import os

# Modeling
import lightgbm as lgb

#######################################################################################################################
# How to use
# 1. Change the status = 'first' if this is the first run of the tuning process, status = 'continue' otherwise.
# 2. Adjust num_trials you wish to run accordingly.
# 3. Modify the data preparation.
# 4. Change param_space if needed.
# 5. Submit the complemented lgbm_optuna.swb file through tmux.

# Note1. This script can be sumbit multiple times to parallel it.
# Note2. Check lgbm_optuna_eval.ipynb out on how to do evalution after the tuning.
#######################################################################################################################\

parser = argparse.ArgumentParser(description="set arguments for LightGBM optuna job")
parser.add_argument(
    "--status", default="first", type=str, help="status of job, first or continue"
)
parser.add_argument("--num_trials", default=90, type=int, help="number of trials")
# parser.add_argument("--study_name", default = "lgbm_mse", type = str, help = "study name")
parser.add_argument(
    "--metric", default="mse", type=str, help="metric for model training"
)
# parser.add_argument(
#     "--data",
#     default="/home/shared/irisk_bop/modeling/train.csv",
#     type=str,
#     help="data location",
# )
parser.add_argument("--capped", default=1e5, type=float, help="capped loss")
parser.add_argument("--coverage", default="BG", type=str, help="type of coverage")
parser.add_argument(
    "--response", default="ObsAnnualizedLoss", type=str, help="response of models"
)
args = parser.parse_args()

# first or continue
# status = 'first'
status = args.status

# number of trials
num_trials = args.num_trials

# Name of the .db
metric = args.metric
coverage = args.coverage
study_name = "lgbm_" + str(num_trials) + "_" + str(coverage) + "_" + str(metric)

# Data preparation
os.chdir("C:/Users/Asus/Desktop/IRisk Lab/Week10&11/SIMTree")
train = pd.read_csv("boston_housing.csv")
y_train = train[[args.response]]
train
train.drop(columns=[], inplace=True)
X_train = train
# X_train = train.loc[train["CoverageCd_" + args.coverage] == 1]
del train

x_train_sparse = sparse.csr_matrix(X_train)

#######################################################################################################################
def objective(trial):
    train_set = lgb.Dataset(x_train_sparse, y_train.values.ravel(), free_raw_data=False)

    param_space = {
        "objective": "regression",
        "boosting_type": "gbdt",
        #         'weight_column': 'weight_new',
        "verbosity": -1,
        "subsample": trial.suggest_uniform("subsample", 0.5, 1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 30, 120, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 35, 1),
        "subsample_for_bin": trial.suggest_int(
            "subsample_for_bin", 20000, 300000, 20000
        ),
        "min_child_samples": trial.suggest_int("min_child_samples", 80, 300, 10),
        "reg_alpha": trial.suggest_uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_uniform("reg_lambda", 0.0, 1.0),
    }

    cv_results = lgb.cv(
        param_space,
        train_set,
        num_boost_round=1000,
        nfold=10,
        early_stopping_rounds=200,
        # mse mae
        metrics=metric,
        # metrics = 'mae',
        stratified=False,
        seed=50,
    )

    if metric == "mse":
        _loss_attribute = "l2-mean"
    elif metric == "mae":
        _loss_attribute = "l1-mean"
    elif metric == "poisson":
        _loss_attribute = "poisson-mean"

    loss = np.min(cv_results[_loss_attribute])
    return loss


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
storage_name = "sqlite:///{}.db".format(study_name)

if status == "first":
    study = optuna.create_study(
        sampler=TPESampler(),
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
    )
else:
    study = optuna.load_study(study_name=study_name, storage=storage_name)

print(f"Total number of trials before: {len(study.trials)} trials.")

study.optimize(objective, n_trials=num_trials)

print(f"Finish runinng {num_trials} trials.")

print(f"Total number of trials after: {len(study.trials)} trials.")
