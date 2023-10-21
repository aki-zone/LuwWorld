import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtWidgets import QLineEdit

from World.Entity.Render.LuwEntity.LPoint import LPoint
from World.Utils.MathLab import Matrix3D
from World.Utils.MathLab.Matrix3D import lookAt


class World:

    def draw_lines(self, painter):
        # 八个空间坐标点
        a1, b1, c1, d1 = LPoint(100, 100, 20), LPoint(150, 100, 20), LPoint(150, 50, 20), LPoint(100, 50, 20)
        a2, b2, c2, d2 = LPoint(100, 100, 70), LPoint(150, 100, 70), LPoint(150, 50, 70), LPoint(100, 50, 70)

        vertices = np.array([
            [-100, -100, -100, 100],
            [100, -100, -100, 100],
            [100, 100, -100, 100],
            [-100, 100, -100, 100],
            [-100, -100, 100, 100],
            [100, -100, 100, 100],
            [100, 100, 100, 100],
            [-100, 100, 100, 100]
        ])

        # 定义相机位置、观察点和上向量
        eye = np.array([0, 50, 50])
        center = np.array([100, 2, 30])
        up = np.array([0, 1, 0])

        # 计算观察矩阵
        Mv = lookAt(eye, center, up)
        print(Mv)

        transformed_vertices = np.dot(vertices, Mv.T)
        print(transformed_vertices)

        # 获取屏幕的宽度和高度
        screen_width = painter.device().width()
        screen_height = painter.device().height()

        # 计算屏幕中心的坐标
        center_x = screen_width // 2
        center_y = screen_height // 2

        # 平移所有顶点坐标，使其相对于屏幕中心绘制
        transformed_vertices[:, 0] += center_x
        transformed_vertices[:, 1] += center_y

        pen = QPen()
        pen.setColor(Qt.black)
        pen.setWidth(5)
        painter.setPen(pen)

        #painter.drawLine(2,5,600,500)
        # 绘制方块的边
        for face in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]:
            for i in range(4):
                v1 = transformed_vertices[face[i]]
                v2 = transformed_vertices[face[(i + 1) % 4]]
                painter.drawLine(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))

