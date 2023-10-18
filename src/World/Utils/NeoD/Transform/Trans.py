from typing import List

import numpy as np
from PyQt5.QtCore import QPoint

from World.Utils.MathLab import Matirx


class Trans2D:
    @staticmethod
    def translate_point(point: QPoint, tx: float, ty: float) -> np.ndarray:
        """
        Translate a QPoint object by (tx, ty).

        Args:
            point (QPoint): The input QPoint to be translated.
            tx (int): The translation along the x-axis.
            ty (int): The translation along the y-axis.

        Returns:
            trans_matrix: The translation matrix.
        """
        matrix = Matirx.Matirx2D.point_to_matrix(point)
        trans_matrix = Matirx.Matirx2D.translate(matrix, tx, ty)
        _temp = Matirx.Matirx2D.extract_point(matrix)
        point.setX(_temp.x())
        point.setY(_temp.y())
        return trans_matrix

    @staticmethod
    def scale_point(point: QPoint, sx: float, sy: float) -> np.ndarray:
        """
        Scale a QPoint object by (tx, ty).

        Args:
            point (QPoint): The input QPoint to be translated.
            sx (int): The Scaling along the x-axis.
            sy (int): The Scaling  along the y-axis.

        Returns:
            trans_matrix: The Scaling  matrix.
        """
        matrix = Matirx.Matirx2D.point_to_matrix(point)
        trans_matrix = Matirx.Matirx2D.scale(matrix, sx, sy)
        _temp = Matirx.Matirx2D.extract_point(matrix)
        point.setX(_temp.x())
        point.setY(_temp.y())
        return trans_matrix

    @staticmethod
    def rotate_point(point: QPoint, angle:float) -> np.ndarray:
        """
        rotate a QPoint object by (tx, ty).

        Args:
            point (QPoint): The input QPoint to be rotated.
            angle (float): The rotation angle in degrees.

        Returns:
            trans_matrix: The rotating  matrix.
        """
        matrix = Matirx.Matirx2D.point_to_matrix(point)
        trans_matrix = Matirx.Matirx2D.rotate(matrix, angle)
        _temp = Matirx.Matirx2D.extract_point(matrix)
        point.setX(_temp.x())
        point.setY(_temp.y())
        return trans_matrix

    @staticmethod
    def shear_point(point: QPoint, shx: float, shy: float) -> np.ndarray:
        """
        shear a QPoint object by (tx, ty).

        Args:
            point (QPoint): The input QPoint to be rotated.
            shx (float): The shear factor along the x-axis.
            shy (float): The shear factor along the y-axis.

        Returns:
            trans_matrix: The rotating  matrix.
        """
        matrix = Matirx.Matirx2D.point_to_matrix(point)
        trans_matrix = Matirx.Matirx2D.shear(matrix, shx, shy)
        _temp = Matirx.Matirx2D.extract_point(matrix)
        point.setX(_temp.x())
        point.setY(_temp.y())
        return trans_matrix

    @staticmethod
    def compose(input_matrix: np.ndarray, transformations: List[np.ndarray]) -> np.ndarray:
        """
        shear a QPoint object by (tx, ty).

        Args:
            input_matrix (ndarray): The input 3x3 matrix to store the composed transformation.
            transformations (List[ndarray]): List of transformation matrices to compose.

        Returns:
            trans_matrix: The rotating  matrix.
        """
        return Matirx.Matirx2D.compose(input_matrix, transformations)

    @staticmethod
    def matrix_mul(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """
        Multiply two 3x3 matrices.
        Args:
            matrix1 (ndarray): The first 3x3 matrix.
            matrix2 (ndarray): The second 3x3 matrix.

        Returns:
            ndarray: The resulting 3x3 matrix.
        """
        return Matirx.Matirx2D.matrix_multiply(matrix1,matrix2)
