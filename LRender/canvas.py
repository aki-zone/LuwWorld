from PIL import Image, ImageColor
import typing as t


class Canvas:
    # __slots__ = "filename", "height", "width"

    # 初始化图像信息
    def __init__(self, filename=None, height=500, width=500):
        self.filename = filename
        self.height, self.width = height, width

        self.img = Image.new("RGBA",
                             (self.height, self.width),
                             (0, 0, 0, 0))

        # pyqt5实现模式
        # self.img = QImage(self.width, self.height, QImage.Format_RGBA8888).fill(QColor(0, 0, 0, 0))

    def draw(self, dots, color: t.Union[tuple, str]):
        if isinstance(color, str):
            # 获取color信息
            color = ImageColor.getrgb(color)
        if isinstance(dots, tuple):
            dots = [dots]

        for dot in dots:
            # 设置透明通道为完全不透明
            self.img.putpixel((int(dot[0]+0.5), int(dot[1]+0.5)), color + (255,))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.img.save(self.filename)
