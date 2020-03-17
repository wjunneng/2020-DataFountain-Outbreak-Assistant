# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

sys.path.append(os.path.abspath('..'))

from . import convert_tf_checkpoint_to_pytorch
