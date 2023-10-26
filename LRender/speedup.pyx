import cython
import numpy as np
cimport numpy as np
cimport cython

# 使用Cpython针对性加速运算

# def表示python可调用函数
# cdef 表示cython内部函数
# cpdef表示cython和python都可以调用的函数

# 调用C-math.h库中的 sqrt函数
cdef extern from "math.h":
    double sqrt(double x)


def normalize(double x, double y, double z):  # 归一化单位向量
    cdef double unit = sqrt(x * x + y * y + z * z)  # 取模长
    if unit == 0:  #防止被除数为0的情况
        return 0, 0, 0
    return x / unit, y / unit, z / unit

cdef (int, int) get_min_max(double a, double b, double c):  #三个值中取最大 和 最小值
    cdef double min_num = a
    cdef double max_num = a
    if min_num > b:
        min_num = b
    if min_num > c:
        min_num = c
    if max_num < b:
        max_num = b
    if max_num < c:
        max_num = c
    return int(min_num), int(max_num)

@cython.boundscheck(False)
cpdef double dot_product(double a0, double a1, double a2, double b0, double b1, double b2):
    # 点乘,由c代码优化
    cdef double result = a0 * b0 + a1 * b1 + a2 * b2
    return result

@cython.boundscheck(False)
cpdef (double, double, double) cross_product(double a0, double a1, double a2, double b0, double b1, double b2):
    # 叉乘运算, 减法以满足右手系坐标法则
    cdef double x = a1 * b2 - a2 * b1
    cdef double y = a2 * b0 - a0 * b2
    cdef double z = a0 * b1 - a1 * b0
    return x, y, z

# 这里还没吃透
@cython.boundscheck(False)
def generate_faces(double [:, :, :] triangles, int width, int height):
    """ draw the triangle faces with z buffer

    Args:
        triangles: groups of vertices

    FYI:
        * zbuffer, https://github.com/ssloy/tinyrenderer/wiki/Lesson-3:-Hidden-faces-removal-(z-buffer)
        * uv mapping and perspective correction
    """
    cdef int i, j, k, length
    cdef double bcy, bcz, x, y, z
    cdef double a[3], b[3], c[3], m[3], bc[3], uva[2], uvb[2], uvc[2]
    cdef int minx, maxx, miny, maxy
    length = triangles.shape[0]
    zbuffer = {}
    faces = []

    for i in range(length):
        a = triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2]
        b = triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2]
        c = triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2]
        uva = triangles[i, 0, 3], triangles[i, 0, 4]
        uvb = triangles[i, 1, 3], triangles[i, 1, 4]
        uvc = triangles[i, 2, 3], triangles[i, 2, 4]
        minx, maxx = get_min_max(a[0], b[0], c[0])
        miny, maxy = get_min_max(a[1], b[1], c[1])
        pixels = []
        for j in range(minx, maxx + 2):
            for k in range(miny - 1, maxy + 2):
                # 必须显式转换成 double 参与底下的运算，不然结果是错的
                x = j
                y = k

                m[0], m[1], m[2] = cross_product(c[0] - a[0], b[0] - a[0], a[0] - x, c[1] - a[1], b[1] - a[1], a[1] - y)
                if abs(m[2]) > 0:
                    bcy = m[1] / m[2]
                    bcz = m[0] / m[2]
                    bc = (1 - bcy - bcz, bcy, bcz)
                else:
                    continue

                # here, -0.00001 because of the precision lose
                if bc[0] < -0.00001 or bc[1] < -0.00001 or bc[2] < -0.00001:
                    continue

                z = 1 / (bc[0] / a[2] + bc[1] / b[2] + bc[2] / c[2])

                # Blender 导出来的 uv 数据，跟之前的顶点数据有一样的问题，Y轴是个反的，
                # 所以这里的纹理图片要旋转一下才能 work
                v = (uva[0] * bc[0] / a[2] + uvb[0] * bc[1] / b[2] + uvc[0] * bc[2] / c[2]) * z * width
                u = height - (uva[1] * bc[0] / a[2] + uvb[1] * bc[1] / b[2] + uvc[1] * bc[2] / c[2]) * z * height

                # https://en.wikipedia.org/wiki/Pairing_function
                idx = ((x + y) * (x + y + 1) + y) / 2
                if zbuffer.get(idx) is None or zbuffer[idx] < z:
                    zbuffer[idx] = z
                    pixels.append((i, j, k, int(u) - 1, int(v) - 1))

        faces.append(pixels)
    return faces
