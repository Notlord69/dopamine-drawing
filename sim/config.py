"""Where the code looks for data and the robot model. Override with environment variables.

  FLYARM_DATA_DIR    data folder: weights_male-cns_v1.0/, shortlist_week1.csv, wsyn_choice.json,
                     week7/, and each notebook's results (week10/, week10b/, ...). Default <repo>/data.
  FLYARM_MENAGERIE   mujoco_menagerie checkout (arm/panda_env.py reads the same variable).
                     Default <repo>/mujoco_menagerie.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('FLYARM_DATA_DIR', REPO_ROOT / 'data'))
MENAGERIE_DIR = Path(os.environ.get('FLYARM_MENAGERIE', REPO_ROOT / 'mujoco_menagerie'))
