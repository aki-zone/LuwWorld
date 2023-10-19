import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPen

from World.Entity.Render.Camera import Camera
from World.Utils.MathLab import Matrix3D
from World.Utils.NeoD.Transform.Trans import Trans2D


class World:

    def draw_lines(self, painter):
        # 八个空间坐标点
        a1, b1, c1, d1 = QPoint(100, 100, 20), QPoint(150, 100, 20), QPoint(150, 50, 20), QPoint(100, 50, 20)
        a2, b2, c2, d2 = QPoint(100, 100, 70), QPoint(150, 100, 70), QPoint(150, 50, 70), QPoint(100, 50, 70)

        c_pos = [5, 10, 7]
        c_self = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        Mv = np.array([
            [c_self[0][0], c_self[1][0], c_self[2][0],
             -(c_self[0][0] * c_pos[0] + c_self[1][0] * c_pos[1] + c_self[2][0] * c_pos[2])],
            [c_self[0][1], c_self[1][1], c_self[2][1],
             -(c_self[0][1] * c_pos[0] + c_self[1][1] * c_pos[1] + c_self[2][1] * c_pos[2])],
            [-c_self[0][2], -c_self[1][2], -c_self[2][2],
             +(c_self[0][2] * c_pos[0] + c_self[1][2] * c_pos[1] + c_self[2][2] * c_pos[2])],
            [0, 0, 0, 1]
        ])

        _a1 = Matrix3D.point_to_matrix(a1)
        _b1 = Matrix3D.point_to_matrix(b1)
        _c1 = Matrix3D.point_to_matrix(c1)
        _d1 = Matrix3D.point_to_matrix(d1)
        _a2 = Matrix3D.point_to_matrix(a2)
        _b2 = Matrix3D.point_to_matrix(b2)
        _c2 = Matrix3D.point_to_matrix(c2)
        _d2 = Matrix3D.point_to_matrix(d2)

        a1 = Matrix3D.extract_point(_a1)
        b1 = Matrix3D.extract_point(_b1)
        c1 = Matrix3D.extract_point(_c1)
        d1 = Matrix3D.extract_point(_d1)
        a2 = Matrix3D.extract_point(_a2)
        b2 = Matrix3D.extract_point(_b2)
        c2 = Matrix3D.extract_point(_c2)
        d2 = Matrix3D.extract_point(_d2)

        a1, b1, c1, d1 = (QPoint(a1.getX(), a1.getY()), QPoint(b1.getX(), b1.getY()),
                          QPoint(c1.getX(),c1.getY()), QPoint(d1.getX(), d1.getY()))
        a2, b2, c2, d2 = (QPoint(a2.getX(), a2.getY()), QPoint(b2.getX(), b2.getY()),
                          QPoint(c2.getX(), c2.getY()), QPoint(d2.getX(), d2.getY()))

        pen = QPen()
        pen.setColor(Qt.black)
        pen.setWidth(5)
        painter.setPen(pen)

        painter.drawLine(a1, b1)
        painter.drawLine(b1, c1)
        painter.drawLine(c1, d1)
        painter.drawLine(d1, a2)
        painter.drawLine(a2, b2)
        painter.drawLine(b2, c2)
        painter.drawLine(c2, d2)
        # 添加更多绘制元素的代码
