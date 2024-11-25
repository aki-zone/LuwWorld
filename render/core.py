import typing as t
from functools import partial

import numpy as np
from copy import deepcopy
from .canvas import Canvas

import speedup


# 2D part


class Vec2d:
    # 使用 __slots__ 优化内存，限制实例只能包含 x, y, arr
    __slots__ = "x", "y", "arr"

    def __init__(self, *args):
        """
        初始化二维向量:
        1. 支持从 Vec3d 转换 (取前两个分量)
        2. 支持直接传入两个值
        """
        if len(args) == 1 and isinstance(args[0], Vec3d):
            self.arr = Vec3d.narr  # 从 Vec3d 中获取分量
        else:
            assert len(args) == 2  # 确保传入两个值
            self.arr = list(args)

        # x, y 分量均取整数值
        self.x, self.y = [d if isinstance(d, int) else int(d + 0.5) for d in self.arr]

    def __repr__(self):
        return f"Vec2d({self.x}, {self.y})"

    def __truediv__(self, other):
        """
        计算当前点和 other 点的斜率
        """
        return (self.y - other.y) / (self.x - other.x)

    def __eq__(self, other):
        """
        判断两个点是否相等
        """
        return self.x == other.x and self.y == other.y


def draw_line(
    v1: Vec2d, v2: Vec2d, canvas: Canvas, color: t.Union[tuple, str] = "white"
):
    """
    用指定颜色绘制一条线，基于 Bresenham 算法

    参考：https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    v1, v2 = deepcopy(v1), deepcopy(v2)  # 防止修改原始点
    if v1 == v2:
        canvas.draw((v1.x, v1.y), color=color)  # 如果两个点相同，只绘制一个点
        return

    # 判断斜率是否陡峭
    steep = abs(v1.y - v2.y) > abs(v1.x - v2.x)
    if steep:  # 如果斜率陡峭，交换 x 和 y
        v1.x, v1.y = v1.y, v1.x
        v2.x, v2.y = v2.y, v2.x
    v1, v2 = (v1, v2) if v1.x < v2.x else (v2, v1)  # 确保 v1 在左边
    slope = abs((v1.y - v2.y) / (v1.x - v2.x))  # 计算斜率
    y = v1.y
    error: float = 0  # 初始化误差
    incr = 1 if v1.y < v2.y else -1  # y 的增量方向
    dots = []
    for x in range(int(v1.x), int(v2.x + 0.5)):
        dots.append((int(y), x) if steep else (x, int(y)))  # 记录像素点
        error += slope
        if abs(error) >= 0.5:  # 如果误差超过 0.5，调整 y 值
            y += incr
            error -= 1

    canvas.draw(dots, color=color)  # 绘制计算出的像素点


def draw_triangle(v1, v2, v3, canvas, color, wireframe=False):
    """
    绘制一个三角形，可以选择绘制线框或者填充

    参考：http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    """
    _draw_line = partial(draw_line, canvas=canvas, color=color)

    if wireframe:
        # 绘制三角形的三条边
        _draw_line(v1, v2)
        _draw_line(v2, v3)
        _draw_line(v1, v3)
        return

    def sort_vertices_asc_by_y(vertices):
        """
        按 y 坐标升序排序顶点
        """
        return sorted(vertices, key=lambda v: v.y)

    def fill_bottom_flat_triangle(v1, v2, v3):
        """
        填充底边平坦的三角形
        """
        invslope1 = (v2.x - v1.x) / (v2.y - v1.y)  # 左边斜率
        invslope2 = (v3.x - v1.x) / (v3.y - v1.y)  # 右边斜率

        x1 = x2 = v1.x
        y = v1.y

        while y <= v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))  # 填充水平线
            x1 += invslope1
            x2 += invslope2
            y += 1

    def fill_top_flat_triangle(v1, v2, v3):
        """
        填充顶边平坦的三角形
        """
        invslope1 = (v3.x - v1.x) / (v3.y - v1.y)  # 左边斜率
        invslope2 = (v3.x - v2.x) / (v3.y - v2.y)  # 右边斜率

        x1 = x2 = v3.x
        y = v3.y

        while y > v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))  # 填充水平线
            x1 -= invslope1
            x2 -= invslope2
            y -= 1

    v1, v2, v3 = sort_vertices_asc_by_y((v1, v2, v3))

    # 填充不同类型的三角形
    if v1.y == v2.y == v3.y:  # 三个顶点 y 值相同，跳过
        pass
    elif v2.y == v3.y:  # 顶边平坦
        fill_bottom_flat_triangle(v1, v2, v3)
    elif v1.y == v2.y:  # 底边平坦
        fill_top_flat_triangle(v1, v2, v3)
    else:
        # 将非平坦三角形拆分为两个平坦三角形
        v4 = Vec2d(int(v1.x + (v2.y - v1.y) / (v3.y - v1.y) * (v3.x - v1.x)), v2.y)
        fill_bottom_flat_triangle(v1, v2, v4)
        fill_top_flat_triangle(v2, v4, v3)

# 3D 部分
class Vec3d:
    __slots__ = "x", "y", "z", "arr"

    def __init__(self, *args):
        """
        初始化三维向量:
        1. 从 Vec4d 转换
        2. 或直接传入三个值
        """
        if len(args) == 1 and isinstance(args[0], Vec4d):
            vec4 = args[0]
            arr_value = (vec4.x, vec4.y, vec4.z)
        else:
            assert len(args) == 3
            arr_value = args
        self.arr = np.array(arr_value, dtype=np.float)
        self.x, self.y, self.z = self.arr

    def __repr__(self):
        return repr(f"Vec3d({','.join([repr(d) for d in self.arr])})")

    def __sub__(self, other):
        """
        向量减法
        """
        return self.__class__(*[ds - do for ds, do in zip(self.arr, other.arr)])

    def __bool__(self):
        """
        判断是否为零向量
        """
        return any(self.arr)


class Mat4d:
    """
    四维矩阵，支持基本矩阵运算
    """
    def __init__(self, narr=None, value=None):
        self.value = np.matrix(narr) if value is None else value

    def __repr__(self):
        return repr(self.value)

    def __mul__(self, other):
        """
        矩阵相乘
        """
        return self.__class__(value=self.value * other.value)


class Vec4d(Mat4d):
    """
    四维向量，继承自 Mat4d
    """
    def __init__(self, *narr, value=None):
        if value is not None:
            self.value = value
        elif len(narr) == 1 and isinstance(narr[0], Mat4d):
            self.value = narr[0].value
        else:
            assert len(narr) == 4
            self.value = np.matrix([[d] for d in narr])

        self.x, self.y, self.z, self.w = (
            self.value[0, 0],
            self.value[1, 0],
            self.value[2, 0],
            self.value[3, 0],
        )

# Math util
def normalize(v: Vec3d):
    return Vec3d(*speedup.normalize(*v.arr))


def dot_product(a: Vec3d, b: Vec3d):
    return speedup.dot_product(*a.arr, *b.arr)


def cross_product(a: Vec3d, b: Vec3d):
    return Vec3d(*speedup.cross_product(*a.arr, *b.arr))


def get_light_intensity(face) -> float:
    light = Vec3d(-2, 4, -10)
    v1, v2, v3 = face
    up = normalize(cross_product(v2 - v1, v3 - v1))
    return dot_product(up, normalize(light))


def look_at(eye: Vec3d, target: Vec3d, up: Vec3d = Vec3d(0, -1, 0)) -> Mat4d:
    """
    http://www.songho.ca/opengl/gl_camera.html#lookat

    Args:
        eye: 摄像机的世界坐标位置
        target: 观察点的位置
        up: 就是你想让摄像机立在哪个方向
            https://stackoverflow.com/questions/10635947/what-exactly-is-the-up-vector-in-opengls-lookat-function
            这里默认使用了 0, -1, 0， 因为 blender 导出来的模型数据似乎有问题，导致y轴总是反的，于是把摄像机的up也翻一下得了。
    """
    f = normalize(eye - target)
    l = normalize(cross_product(up, f))  # noqa: E741
    u = cross_product(f, l)

    rotate_matrix = Mat4d(
        [[l.x, l.y, l.z, 0], [u.x, u.y, u.z, 0], [f.x, f.y, f.z, 0], [0, 0, 0, 1.0]]
    )
    translate_matrix = Mat4d(
        [[1, 0, 0, -eye.x], [0, 1, 0, -eye.y], [0, 0, 1, -eye.z], [0, 0, 0, 1.0]]
    )

    return Mat4d(value=(rotate_matrix * translate_matrix).value)


def perspective_project(r, t, n, f, b=None, l=None):  # noqa: E741
    """
    目的：
        把相机坐标转换成投影在视网膜的范围在(-1, 1)的笛卡尔坐标

    原理：
        对于x，y坐标，相似三角形可以算出投影点的x，y
        对于z坐标，是假设了near是-1，far是1，然后带进去算的
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix

    推导出来的矩阵：
        [
            2n/(r-l) 0        (r+l/r-l)   0
            0        2n/(t-b) (t+b)/(t-b) 0
            0        0        -(f+n)/f-n  (-2*f*n)/(f-n)
            0        0        -1          0
        ]

    实际上由于我们用的视网膜(near pane)是个关于远点对称的矩形，所以矩阵简化为：
        [
            n/r      0        0           0
            0        n/t      0           0
            0        0        -(f+n)/f-n  (-2*f*n)/(f-n)
            0        0        -1          0
        ]

    Args:
        r: right, t: top, n: near, f: far, b: bottom, l: left
    """
    return Mat4d(
        [
            [n / r, 0, 0, 0],
            [0, n / t, 0, 0],
            [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    )


def draw(screen_vertices, world_vertices, model, canvas, wireframe=True):
    """standard algorithm
    """
    for triangle_indices in model.indices:
        vertex_group = [screen_vertices[idx - 1] for idx in triangle_indices]
        face = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        if wireframe:
            draw_triangle(*vertex_group, canvas=canvas, color="black", wireframe=True)
        else:
            intensity = get_light_intensity(face)
            if intensity > 0:
                draw_triangle(
                    *vertex_group, canvas=canvas, color=(int(intensity * 255),) * 3
                )


def draw_with_z_buffer(screen_vertices, world_vertices, model, canvas):
    """ z-buffer algorithm
    """
    intensities = []
    triangles = []
    for i, triangle_indices in enumerate(model.indices):
        screen_triangle = [screen_vertices[idx - 1] for idx in triangle_indices]
        uv_triangle = [model.uv_vertices[idx - 1] for idx in model.uv_indices[i]]
        world_triangle = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        intensities.append(abs(get_light_intensity(world_triangle)))
        # take off the class to let Cython work
        triangles.append(
            [np.append(screen_triangle[i].arr, uv_triangle[i]) for i in range(3)]
        )

    faces = speedup.generate_faces(
        np.array(triangles, dtype=np.float), model.texture_width, model.texture_height
    )
    for face_dots in faces:
        for dot in face_dots:
            intensity = intensities[dot[0]]
            u, v = dot[3], dot[4]
            color = model.texture_array[u, v]
            canvas.draw((dot[1], dot[2]), tuple(int(c * intensity) for c in color[:3]))
            # TODO: add object rendering mode (no texture)
            # canvas.draw((dot[1], dot[2]), (int(255 * intensity),) * 3)


def render(model, height, width, filename, wireframe=False):
    """
    Args:
        model: the Model object
        height: cavas height
        width: cavas width
        picname: picture file name
    """
    model_matrix = Mat4d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # TODO: camera configration
    view_matrix = look_at(Vec3d(-4, -4, 10), Vec3d(0, 0, 0))
    projection_matrix = perspective_project(0.5, 0.5, 3, 1000)

    world_vertices = []

    def mvp(v):
        world_vertex = model_matrix * v
        world_vertices.append(Vec4d(world_vertex))
        return projection_matrix * view_matrix * world_vertex

    def ndc(v):
        """
        各个坐标同时除以 w，得到 NDC 坐标
        """
        v = v.value
        w = v[3, 0]
        x, y, z = v[0, 0] / w, v[1, 0] / w, v[2, 0] / w
        return Mat4d([[x], [y], [z], [1 / w]])

    def viewport(v):
        x = y = 0
        w, h = width, height
        n, f = 0.3, 1000
        return Vec3d(
            w * 0.5 * v.value[0, 0] + x + w * 0.5,
            h * 0.5 * v.value[1, 0] + y + h * 0.5,
            0.5 * (f - n) * v.value[2, 0] + 0.5 * (f + n),
        )

    # the render pipeline
    screen_vertices = [viewport(ndc(mvp(v))) for v in model.vertices]

    with Canvas(filename, height, width) as canvas:
        if wireframe:
            draw(screen_vertices, world_vertices, model, canvas)
        else:
            draw_with_z_buffer(screen_vertices, world_vertices, model, canvas)
