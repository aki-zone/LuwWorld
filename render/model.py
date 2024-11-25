import numpy
from PIL import Image
from .core import Vec4d

class Model:
    def __init__(self, filename, texture_filename):
        """
        初始化Model类，从OBJ文件和纹理文件中读取数据。

        :param filename: OBJ文件的路径。
        :param texture_filename: 纹理文件的路径。
        """
        self.vertices = []  # 存储顶点数据
        self.uv_vertices = []  # 存储UV坐标数据
        self.uv_indices = []  # 存储UV坐标索引
        self.indices = []  # 存储顶点索引

        # 加载纹理文件
        texture = Image.open(texture_filename)
        self.texture_array = numpy.array(texture)  # 将纹理转换为numpy数组
        self.texture_width, self.texture_height = texture.size  # 获取纹理的宽和高

        # 读取OBJ文件
        with open(filename) as f:
            for line in f:
                # 处理顶点数据行
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")]
                    self.vertices.append(Vec4d(x, y, z, 1))  # 添加顶点到列表中
                # 处理UV坐标数据行
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")]
                    self.uv_vertices.append([u, v])  # 添加UV坐标到列表中
                # 处理面数据行
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0]) for d in facet])  # 添加顶点索引到列表中
                    self.uv_indices.append([int(d[1]) for d in facet])  # 添加UV坐标索引到列表中