# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# io.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import cv2
import json
import pickle
import numpy as np


class IO:
    r"""
    Wrapping io in openlanev2,
    can be modified for different file systems.
    
    """
    def __init__(self) -> None:
        pass

    def os_listdir(self, path : str) -> list:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        list

        """
        return os.listdir(path) 

    def cv2_imread(self, path : str) -> np.ndarray:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        np.ndarray

        """
        return cv2.imread(path) 

    def json_load(self, path : str) -> dict:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        dict

        """
        with open(path, 'r') as f:
            result = json.load(f)
        return result

    def pickle_dump(self, path : str, obj : object) -> None:
        r"""
        Parameters
        ----------
        path : str
        obj : object

        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def pickle_load(self, path : str) -> object:
        r"""
        Parameters
        ----------
        path : str

        Returns
        -------
        object

        """
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

io = IO()
