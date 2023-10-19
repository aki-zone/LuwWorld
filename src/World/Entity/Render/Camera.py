from typing import List

import numpy as np


class Camera:
    def __init__(self,
                 world_position: List = None,
                 self_matrix: np.ndarray = None,
                 view_matrix: np.ndarray = None):
        """
        Initialize a Camera object.
        Args:
            world_position (list or None): Camera's world coordinates [x, y, z].
            self_matrix (list of lists or None): Self matrix for camera transformation.
            view_matrix (list of lists or None): View matrix for camera transformation.
        """
        self.world_position = world_position if world_position is not None else [0, 0, 0]
        self.self_matrix = self_matrix if self_matrix is not None else np.identity(3)
        self.view_matrix = view_matrix

    def set_view_matrix(self, view_matrix):
        """
        Set the view matrix for the camera.

        Args:
            view_matrix (list of lists): The view matrix for camera transformation.
        """
        self.view_matrix = view_matrix
