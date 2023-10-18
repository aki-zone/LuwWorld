from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
import sys


class KeyMonitor(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.key_pressed = set()

    def initUI(self):
        self.setWindowTitle('Key Monitor')
        self.setGeometry(100, 100, 400, 300)
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        if event.type() == Qt.KeyPress:
            self.key_pressed.add(key)
            self.printKeys()
        elif event.type() == Qt.KeyRelease:
            self.key_pressed.discard(key)
            self.printKeys()

    def printKeys(self):
        print(self.key_pressed)
