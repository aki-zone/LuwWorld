from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget, QApplication


class LinePainter:
    def __init__(self, window):
        self.main_window = window

    def draw_line(self, start_point, end_point):
        painter = QPainter(self.main_window)
        pen = QPen()
        pen.setColor(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(start_point, end_point)
        print("a")
