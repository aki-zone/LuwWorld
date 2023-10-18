from PyQt5.QtCore import QLine, Qt, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView

from World.UI.World import World
from World.Utils.NeoD.LPainter.Line import LinePainter

"""
main window which is external framework for rendering the entire project
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 设置窗口属性
        self.setWindowTitle('LuwWorld')
        self.setGeometry(100, 100, 800, 600)

    def paintEvent(self, event):
        painter = QPainter(self)
        world = World()
        world.draw_lines(painter)
