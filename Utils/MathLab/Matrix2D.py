from typing import List

import numpy as np
from PyQt5.QtCore import QPoint


def translate(input_matrix: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """
    Apply 2D translation to the input matrix.

    Args:
        input_matrix (ndarray): The input 3x3 matrix to be translated.
        tx (float): The translation along the x-axis.
        ty (float): The translation along the y-axis.

    Returns:
        ndarray: The translated 3x3 matrix.
    """
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])
    input_matrix[:] = np.dot(translation_matrix, input_matrix)
    return translation_matrix


def scale(input_matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Apply 2D scaling to the input matrix.

    Args:
        input_matrix (ndarray): The input 3x3 matrix to be scaled.
        sx (float): The scaling factor along the x-axis.
        sy (float): The scaling factor along the y-axis.

    Returns:
        ndarray: The scaled 3x3 matrix.
    """
    scaling_matrix = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    input_matrix[:] = np.dot(scaling_matrix, input_matrix)
    return scaling_matrix


def rotate(input_matrix: np.ndarray, angle: float) -> np.ndarray:
    """
    Apply 2D rotation to the input matrix.

    Args:
        input_matrix (ndarray): The input 3x3 matrix to be rotated.
        angle (float): The rotation angle in degrees.

    Returns:
        ndarray: The rotated 3x3 matrix.
    """
    radian_angle = np.radians(angle)
    cos_theta = np.cos(radian_angle)
    sin_theta = np.sin(radian_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    input_matrix[:] = np.dot(rotation_matrix, input_matrix)
    return rotation_matrix


def shear(input_matrix: np.ndarray, shx: float, shy: float) -> np.ndarray:
    """
    Apply 2D shear to the input matrix.

    Args:
        input_matrix (ndarray): The input 3x3 matrix to be sheared.
        shx (float): The shear factor along the x-axis.
        shy (float): The shear factor along the y-axis.

    Returns:
        ndarray: The sheared 3x3 matrix.
    """
    shear_matrix = np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ])
    input_matrix[:] = np.dot(shear_matrix, input_matrix)
    return shear_matrix


def compose(input_matrix: np.ndarray, transformations: List[np.ndarray]) -> np.ndarray:
    """
    Compose multiple transformations into a single transformation matrix.

    Args:
        input_matrix (ndarray): The input 3x3 matrix to store the composed transformation.
        transformations (List[ndarray]): List of transformation matrices to compose.

    Returns:
        ndarray: The composed 3x3 transformation matrix.
    """

    result_matrix = input_matrix
    for transform in transformations:
        result_matrix = np.dot(result_matrix, transform)
    return result_matrix


def matrix_multiply(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Multiply two 3x3 matrices.
    Args:
        matrix1 (ndarray): The first 3x3 matrix.
        matrix2 (ndarray): The second 3x3 matrix.

    Returns:
        ndarray: The resulting 3x3 matrix.
    """

    result_matrix = np.dot(matrix1, matrix2)
    return result_matrix


def point_to_matrix(point: QPoint) -> np.ndarray:
    """
    Convert a QPoint object to a 3x1 matrix.

    Args:
        point (QPoint): The input QPoint object.

    Returns:
        ndarray: The 3x1 matrix representing the QPoint.
    """
    if isinstance(point, QPoint):
        return np.array(
            [
                [point.x()],
                [point.y()],
                [1]
            ]
        )
    else:
        raise ValueError("Matrix2D: Input must be a QPoint object")


def extract_point(result_matrix: np.ndarray) -> QPoint:
    """
    Extract a QPoint object from a 3x1 matrix.

    Args:
        result_matrix (ndarray): The input 3x1 matrix.

    Returns:
        QPoint: The extracted QPoint.
    """
    if result_matrix.shape == (3, 1):
        return QPoint(int(result_matrix[0, 0]), int(result_matrix[1, 0]))
    else:
        raise ValueError("Input matrix must be a 3x1 matrix")
