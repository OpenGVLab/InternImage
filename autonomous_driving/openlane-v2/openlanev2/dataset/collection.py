# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# collection.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

from .frame import Frame
from ..io import io


class Collection:
    r"""
    A collection of frames.
    
    """
    def __init__(self, data_root : str, meta_root : str, collection : str) -> None:
        r"""
        Parameters
        ----------
        data_root : str
        meta_root : str
        collection : str
            Name of collection.

        """
        try:
            meta = io.pickle_load(f'{meta_root}/{collection}.pkl')
        except FileNotFoundError:
            raise FileNotFoundError('Please run the preprocessing first to generate pickle file of the collection.')

        self.frames = {k: Frame(data_root, v) for k, v in meta.items()}
        self.keys = list(self.frames.keys())

    def get_frame_via_identifier(self, identifier : tuple) -> Frame:
        r"""
        Returns a frame with the given identifier (split, segment_id, timestamp).

        Parameters
        ----------
        identifier : tuple
            (split, segment_id, timestamp).

        Returns
        -------
        Frame
            A frame identified by the identifier.

        """
        return self.frames[identifier]

    def get_frame_via_index(self, index : int) -> (tuple, Frame):
        r"""
        Returns a frame with the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        (tuple, Frame)
            The identifier of the frame and the frame.

        """
        return self.keys[index], self.frames[self.keys[index]]
