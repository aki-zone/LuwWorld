from PyQt5.QtCore import QLine, Qt, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView

from World.UI.World import World
from World.Utils.NeoD.LPainter.Line import LinePainter

"""
主要窗口,是整个项目渲染时的外部框架
"""


class MainWindow(QMainWindow):

    # 构造函数
    def __init__(self):
        super().__init__()
        # 设置窗口属性
        self.setWindowTitle('LuwWorld')
        self.setGeometry(100, 100, 800, 600)

    def paintEvent(self, event):
        painter = QPainter(self)
        world = World()
        world.draw_lines(painter)

