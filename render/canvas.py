import typing as t

from PIL import Image, ImageColor

class Canvas:
    def __init__(self, filename=None, height=500, width=500):
        """
        初始化Canvas类，创建一个画布。

        :param filename: 保存画布的文件名，默认为None，不保存文件。
        :param height: 画布的高度，默认为500像素。
        :param width: 画布的宽度，默认为500像素。
        """
        self.filename = filename  # 保存画布的文件名
        self.height, self.width = height, width  # 画布的高和宽
        self.img = Image.new("RGBA", (self.height, self.width), (0, 0, 0, 0))  # 创建一个新的透明画布

    def draw(self, dots, color: t.Union[tuple, str]):
        """
        在画布上绘制点或像素。

        :param dots: 要绘制的点或点的列表，每个点是一个(x, y)的元组。
        :param color: 绘制点的颜色，可以是RGB元组或颜色名称的字符串。
        """
        if isinstance(color, str):  # 如果颜色是字符串，转换为RGB元组
            color = ImageColor.getrgb(color)
        if isinstance(dots, tuple):  # 如果dots是单个点，转换为列表
            dots = [dots]
        for dot in dots:  # 遍历每个点
            self.img.putpixel(dot, color + (255,))  # 在画布上绘制点，颜色加透明度255

    def __enter__(self):
        """
        上下文管理器的进入方法，用于创建Canvas实例并返回。
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        上下文管理器的退出方法，用于保存画布为图像文件。

        :param type: 异常类型。
        :param value: 异常值。
        :param traceback:  traceback对象。
        """
        if self.filename:  # 如果指定了文件名，则保存画布
            self.img.save(self.filename)  # 保存画布为图像文件