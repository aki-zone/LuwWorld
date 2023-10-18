import numpy as np

# 观察者的位置
eye = np.array([0, 0, 5])

# 观察点
target = np.array([0, 0, 0])

# 上方向
up = np.array([0, 1, 0])

# 创建视图矩阵
def create_view_matrix(eye, target, up):
    # 计算观察矩阵
    z_axis = (eye - target) / np.linalg.norm(eye - target)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    translation = -eye
    view_matrix = np.array([
        [x_axis[0], x_axis[1], x_axis[2], np.dot(x_axis, translation)],
        [y_axis[0], y_axis[1], y_axis[2], np.dot(y_axis, translation)],
        [z_axis[0], z_axis[1], z_axis[2], np.dot(z_axis, translation)],
        [0, 0, 0, 1]
    ])
    return view_matrix

# 对场景中的每个对象执行视图变换
def apply_view_transform(model_matrix, view_matrix):
    # 将模型矩阵与视图矩阵相乘
    return np.dot(view_matrix, model_matrix)
