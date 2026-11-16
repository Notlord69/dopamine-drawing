"""Where the code looks for data and the robot model. Override with environment variables.

  DD_DATA_DIR    data folder holding weights_male-cns_v1.0/ (scripts/fetch_connectome.py).
                 Default <repo>/data. An existing copy from fly-connectome-arm works as is.
  DD_MENAGERIE   mujoco_menagerie checkout (scripts/fetch_panda_model.sh). Default <repo>/mujoco_menagerie.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('DD_DATA_DIR', REPO_ROOT / 'data'))
MENAGERIE_DIR = Path(os.environ.get('DD_MENAGERIE', REPO_ROOT / 'mujoco_menagerie'))
WEIGHTS_DIR = DATA_DIR / 'weights_male-cns_v1.0'
