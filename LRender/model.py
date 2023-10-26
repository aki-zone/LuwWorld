import numpy
from PIL import Image
from .core import Vec4d


class Model:
    def __init__(self, filename, texture_filename):

        # 分别为 单顶点坐标数组 和 单纹理坐标数组
        self.vertices = []
        self.uv_vertices = []

        # 分别为 三角顶点索引清单 和 三角纹理索引清单
        self.indices = []
        self.uv_indices = []

        texture = Image.open(texture_filename)
        self.texture_array = numpy.array(texture)
        self.texture_width, self.texture_height = texture.size

        with open(filename) as f:
            for line in f:

                # 以v开头,意为模型顶点坐标信息, 去除空格和v后赋值
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")]
                    self.vertices.append(Vec4d(x, y, z, 1))

                # 以vt开头,意为纹理贴图坐标
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")]
                    self.uv_vertices.append([u, v])

                # 以f开头,存储三个点的索引信息,即在xx数组里的下标,每个点的信息格式形如 a/b/c:
                # a代表该处顶点列表里的下标, b代表该处纹理列表里的下标, c代表该处在法线向量列表里的下标
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0]) for d in facet])
                    self.uv_indices.append([int(d[1]) for d in facet])
