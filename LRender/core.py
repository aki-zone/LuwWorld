from copy import deepcopy
from functools import partial

import numpy as np
import speedup
import typing as t

from LRender.canvas import Canvas


# import Canvas


class Vec2d:
    # 预静态的构造默认参数,减少内存调用
    __slots__ = "x", "y", "arr"

    def __init__(self, *args):
        # Vec3d 情况
        if len(args) == 1 and isinstance(args[0], Vec3d):
            self.arr = Vec3d.arr
        else:
            # 阻塞检测,若出现大于两个元素的输入,则报错
            assert len(args) == 2
            self.arr = list(args)

        # 此处将仅赋值前两个变量值
        self.x, self.y = [_d
                          if isinstance(_d, int)
                          else int(_d + 0.5)
                          for _d in self.arr]

    # 魔术方法:输出字符串
    def __repr__(self):
        return f"Vec2d({self.x}, {self.y})"

    # 魔术方法: 双点斜率
    def __truediv__(self, other):
        return (self.y - other.y) / (self.x - other.x)

    # 魔术方法: 全等
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Vec3d:
    # 预静态的构造默认参数,减少内存调用
    __slots__ = "x", "y", "z", "arr"

    def __init__(self, *args):
        # for Vec4d cast
        if len(args) == 1 and isinstance(args[0], Vec4d):
            vec4 = args[0]
            arr_value = (vec4.x, vec4.y, vec4.z)
        else:
            assert len(args) == 3
            arr_value = args
        self.arr = np.array(arr_value, dtype=np.double)
        self.x, self.y, self.z = self.arr

    def __repr__(self):
        return repr(f"Vec3d({','.join([repr(_d) for _d in self.arr])})")

    # 减法,使用分离元素*初始化
    def __sub__(self, other):
        return self.__class__(*[_d_self - _d_other
                                for _d_self, _d_other in zip(self.arr, other.arr)])

    def __bool__(self):
        """
        vector (0, 0, 0) 即返回 False
        """
        return any(self.arr)

    #
    def __eq__(self, other):
        """
        魔术方法: 全等
        :param other: Vec3d
        :return: Bool
        """
        return (self.x == other.x
                and self.y == other.y
                and self.z == other.z)


class Vec4d:
    def __init__(self, *narr, value=None):

        # 优先使用value赋值
        if value is not None:
            self.value = value

        # 当赋值量为单变量且类型为Mat4d时
        elif len(narr) == 1 and isinstance(narr[0], Mat4d):
            self.value = narr[0].value

        # 四维数值赋值
        else:
            assert len(narr) == 4
            # d = float(narr[0])
            self.value = np.matrix([[d] for d in narr])

        self.x, self.y, self.z, self.w = (
            self.value[0, 0],
            self.value[1, 0],
            self.value[2, 0],
            self.value[3, 0],
        )

        # 构造为一维四列数组, 参数为一个元组所以需要两层括号
        self.arr = self.value.reshape((1, 4))


class Mat4d:
    def __init__(self, narr=None, value=None):
        self.value = np.matrix(narr) if value is None else value

    def __repr__(self):
        return repr(f"Mat4d: {self.value}")

    def __mul__(self, other):
        return self.__class__(value=self.value * other.value)


# 运算工具包装
def normalize(vec: Vec3d):
    return Vec3d(*speedup.normalize(*vec.arr))


def dot_product(vec1: Vec3d, vec2: Vec3d):
    return speedup.dot_product(*vec1.arr, *vec2.arr)


def cross_product(vec1: Vec3d, vec2: Vec3d):
    return Vec3d(*speedup.cross_product(*vec1.arr, *vec2.arr))


def look_at(eye: Vec3d, target: Vec3d, up: Vec3d = Vec3d(0, 1, 0)) -> Mat4d:
    """
    Args:
        eye: 摄像机的世界坐标位置
        target: 观察点的位置
        up: 就是你想让摄像机立在哪个方向
    """

    # 归一化获取前向量
    front = normalize(eye - target)

    # 叉乘计算左向量
    left = normalize(cross_product(up, front))

    # 叉乘重定义上向量
    upon = cross_product(front, left)

    # 旋转矩阵
    rotate_mat = Mat4d(
        [
            [left.x, left.y, left.z, 0],
            [upon.x, upon.y, upon.z, 0],
            [front.x, front.y, front.z, 0],
            [0, 0, 0, 1.0]
        ]
    )

    # 位移矩阵
    translate_mat = Mat4d(
        [
            [1, 0, 0, -eye.x],
            [0, 1, 0, -eye.y],
            [0, 0, 1, -eye.z],
            [0, 0, 0, 1.0]
        ]
    )

    return Mat4d(value=(rotate_mat * translate_mat).value)


# 贝斯曼直线绘制法
def draw_line(
        v1: Vec2d, v2: Vec2d, canv: Canvas, color: t.Union[tuple, str] = "white"
):
    # 调用深拷贝,即引用参变
    v1, v2 = deepcopy(v1), deepcopy(v2)

    if v1 == v2:
        canv.draw((v1.x, v1.y), color=color)
        return

    # steep信息参数,确立主方向
    steep = abs(v1.y - v2.y) > abs(v1.x - v2.x)

    # 主方向为y, 则x,y轴信息交换
    if steep:
        v1.x, v1.y = v1.y, v1.x
        v2.x, v2.y = v2.y, v2.x

    # 延伸方向为负轴则交换二点
    v1, v2 = (v1, v2) if v1.x < v2.x else (v2, v1)

    # slope斜率值,计算二点斜率
    slope = abs((v1.y - v2.y) / (v1.x - v2.x))

    # y 为直线绘制纵坐标起始自增量
    y = v1.y

    """
    error : 误差量,为float类型
    incr : 步进量
    dots : 结果点集
    
    公式为:
        y = y if 
          = y
    """

    error: float = 0
    incr = 1 if v1.y < v2.y else -1
    dots = []

    for x in range(int(v1.x), int(v2.x + 0.5)):

        # 交换x,y轴后, 记录值需记录回原值,即再交换一次
        dots.append((int(y), x) if steep else (x, int(y)))

        error += slope
        if abs(error) >= 0.5:
            y += incr
            error -= 1

    canv.draw(dots, color=color)


def draw_triangle(v1: Vec2d, v2: Vec2d, v3: Vec2d, canvas, color, wireframe=False):
    """
    Draw a triangle with 3 ordered vertices

    http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    """
    _draw_line = partial(draw_line, canvas=canvas, color=color)

    # 需要线框, 则绘制三角形边
    if wireframe:
        _draw_line(v1, v2)
        _draw_line(v2, v3)
        _draw_line(v1, v3)
        return

    # 点阵排序,按照每个点的y值排序
    def sort_vertices_asc_by_y(vertices):
        return sorted(vertices, key=lambda v: v.y)

    # 等顶三角形, 即vec2 = vec3
    def fill_bottom_flat_triangle(vec1, vec2, vec3):

        # 左右腰线的斜率倒数
        invslope1 = (vec2.x - vec1.x) / (vec2.y - vec1.y)
        invslope2 = (vec3.x - vec1.x) / (vec3.y - vec1.y)

        x1 = x2 = vec1.x
        y_start = vec1.y

        # 扫描线填充
        while y_start <= vec2.y:
            _draw_line(Vec2d(x1, y_start), Vec2d(x2, y_start))
            x1 += invslope1
            x2 += invslope2
            y_start += 1

    # 等底三角形, 即vec2 = vec1
    def fill_top_flat_triangle(vec1, vec2, vec3):
        invslope1 = (vec3.x - vec1.x) / (vec3.y - vec1.y)
        invslope2 = (vec3.x - vec2.x) / (vec3.y - vec2.y)

        x1 = x2 = vec3.x
        y_start = vec3.y

        while y_start > vec2.y:
            _draw_line(Vec2d(x1, y_start), Vec2d(x2, y_start))
            x1 -= invslope1
            x2 -= invslope2
            y_start -= 1

    # 排序,按照y坐标 vec1 < vec2 < vec3
    v1, v2, v3 = sort_vertices_asc_by_y((v1, v2, v3))

    # 填充
    # 全平行, 无需填充
    if v1.y == v2.y == v3.y:
        pass

    # 等底/等顶三角形, 直接调用函数填充
    elif v2.y == v3.y:
        fill_bottom_flat_triangle(v1, v2, v3)
    elif v1.y == v2.y:
        fill_top_flat_triangle(v1, v2, v3)

    # 以vec2水平轴为中界线, 切分为上下两个三角形
    else:
        v4 = Vec2d(int(v1.x + (v2.y - v1.y) / (v3.y - v1.y) * (v3.x - v1.x)), v2.y)
        fill_bottom_flat_triangle(v1, v2, v4)
        fill_top_flat_triangle(v2, v4, v3)


###############################################
# 暂时没有进行矩阵推导,所以没有额外解释.矩阵结果原封照用
def perspective_project(r, t, n, f, b=None, l=None):  # noqa: E741
    """
    目的：
        把相机坐标转换成投影在视网膜的范围在(-1, 1)的笛卡尔坐标

    原理：
        对于x，y坐标，相似三角形可以算出投影点的x，y
        对于z坐标，是假设了near是-1，far是1，然后带进去算的
        https://www.songho.ca/opengl/gl_projectionmatrix.html
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


def get_light_intensity(face) -> float:
    # light: 光源位
    light = Vec3d(-2, 4, -10)
    v1, v2, v3 = face

    # 将三点曲面近似为三角形, 两个边向量的叉乘为三角形的面法向量,并归一化
    up = normalize(cross_product(v2 - v1, v3 - v1))

    # 方向点积,当重合角度最小时, 点积结果最大,同时光照最强
    return dot_product(up, normalize(light))


def draw(screen_vertices, world_vertices, model, canv, wireframe=True):
    # 遍历模型里的顶点索引清单
    for triangle_indices in model.indices:
        vertex_group = [screen_vertices[idx - 1] for idx in triangle_indices]
        face = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]

        # 绘制线框
        if wireframe:
            draw_triangle(*vertex_group, canvas=canv, color="black", wireframe=True)
        else:
            intensity = get_light_intensity(face)
            if intensity > 0:
                draw_triangle(
                    *vertex_group, canvas=canv, color=(int(intensity * 255),) * 3
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
        np.array(triangles, dtype=np.double), model.texture_width, model.texture_height
    )
    for face_dots in faces:
        for dot in face_dots:
            intensity = intensities[dot[0]]
            u, v = dot[3], dot[4]
            color = model.texture_array[u, v]
            canvas.draw((dot[1], dot[2]), tuple(int(c * intensity) for c in color[:3]))
            # canv.draw((dot[1], dot[2]), (int(255 * intensity),) * 3)


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

        if w != 0:
            x, y, z = v[0, 0] / w, v[1, 0] / w, v[2, 0] / w
            return Mat4d([[x], [y], [z], [1 / w]])
        else:
            x, y, z = v[0, 0] / 1, v[1, 0] / 1, v[2, 0] / 1
            return Mat4d([[x], [y], [z], [1 / 1]])

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

    with Canvas(filename, height, width) as canv:
        if wireframe:
            draw(screen_vertices, world_vertices, model, canv)
        else:
            draw_with_z_buffer(screen_vertices, world_vertices, model, canv)
