from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPen

from World.Utils.MathLab import Matirx
from World.Utils.NeoD.Transform.Trans import Trans2D


class World:

    def draw_lines(self, painter):
        a, b, c, d = QPoint(100, 100), QPoint(150, 100), QPoint(150, 50), QPoint(100, 50)
        pen = QPen()
        pen.setColor(Qt.black)
        pen.setWidth(5)
        painter.setPen(pen)

        trans = Trans2D()
        tx = 300
        ty = 300

        trans.translate_point(a, tx, ty)
        trans.translate_point(b, tx, ty)
        trans.translate_point(c, tx, ty)
        trans.translate_point(d, tx, ty)

        painter.drawLine(a, b)
        painter.drawLine(b, c)
        painter.drawLine(c, d)
        painter.drawLine(d, a)
        # 添加更多绘制元素的代码
