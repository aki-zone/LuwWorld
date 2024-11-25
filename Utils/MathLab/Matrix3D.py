from typing import List

import numpy as np

from World.Entity.Render.LuwEntity.LPoint import LPoint


def translate(input_matrix: np.ndarray, tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Apply 2D translation to the input matrix.

    Args:
        input_matrix (ndarray): The input 4x4 matrix to be translated.
        tx (float): The translation along the x-axis.
        ty (float): The translation along the y-axis.
        tz (float): The translation along the z-axis.

    Returns:
        ndarray: The translated 3x3 matrix.
    """
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ])
    input_matrix[:] = np.dot(translation_matrix, input_matrix)
    return translation_matrix


def lookAt(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Calculate the view matrix.

    Parameters:
    eye (numpy.ndarray): Camera position (3D vector).
    center (numpy.ndarray): Position of the point being observed, the target where the camera is looking (3D vector).
    up (numpy.ndarray): Camera's up vector (3D vector).

    Returns:
    numpy.ndarray: The view matrix used to transform objects from world coordinates to camera coordinates.
    """

    forward = (center - eye)  # The view vector
    forward = forward.astype(float) / np.linalg.norm(forward).astype(float)  # Vector normalization
    right = np.cross(up, forward)  # Cross product of up and forward vectors, obtaining a right direction vector
    right /= np.linalg.norm(right).astype(float)  # Vector normalization
    new_up = np.cross(forward, right)  # Ensure orthogonality while normalizing the up vector

    view_matrix = np.array([
        [right[0], new_up[0], -forward[0], 0],
        [right[1], new_up[1], -forward[1], 0],
        [right[2], new_up[2], -forward[2], 0],
        [-np.dot(right, eye), -np.dot(new_up, eye), np.dot(forward, eye), 1]
    ])

    return view_matrix


def compose(input_matrix: np.ndarray, transformations: List[np.ndarray]) -> np.ndarray:
    """
    Compose multiple transformations into a single transformation matrix.

    Args:
        input_matrix (ndarray): The input 4x4 matrix to store the composed transformation.
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


def point_to_matrix(point: LPoint) -> np.ndarray:
    """
    Convert a LPoint object to a 3x1 matrix.

    Args:
        point (LPoint): The input LPoint object.

    Returns:
        ndarray: The 3x1 matrix representing the LPoint.
    """
    if isinstance(point, LPoint):
        return np.array(
            [
                point.getX(),
                point.getY(),
                point.getZ(),
                1
            ]
        )
    else:
        raise ValueError("Matrix2D: Input must be a LPoint object")


def extract_point(result_matrix: np.ndarray) -> LPoint:
    """
    Extract a LPoint object from a 3x1 matrix.

    Args:
        result_matrix (ndarray): The input 3x1 matrix.

    Returns:
        LPoint: The extracted LPoint.
    """
    if result_matrix.shape == (4, 1):
        return LPoint(int(result_matrix[0, 0]), int(result_matrix[1, 0]), int(result_matrix[2, 0]))
    else:
        raise ValueError("Input matrix must be a 3x1 matrix")
