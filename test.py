import taichi as ti
##初始化
# Initialize Taichi and run it on CPU (default)
# - `arch=ti.gpu`: Run Taichi on GPU and has Taichi automatically detect the suitable backend
# - `arch=ti.cuda`: For the NVIDIA CUDA backend
# - `arch=ti.metal`: [macOS] For the Apple Metal backend
# - `arch=ti.opengl`: For the OpenGL backend
# - `arch=ti.vulkan`: For the Vulkan backend
# - `arch=ti.dx11`: For the DX11 backend
#ti.init(arch=ti.cpu)
ti.init(arch=ti.gpu)
# Taichi allocates 1 GB GPU memory for field storage by default.
# ti.init(arch=ti.cuda, device_memory_GB=3.4
# ti.init(arch=ti.cuda, device_memory_fraction=0.3)

##field设置
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

# Nested kernels are not supported.
# Nested functions are supported.
# Recursive functions are not supported for now.


#只有在 @ti.kernel and @ti.func 里的是taichi语言 其他是python
# Taichi kernels must be called from the Python-scope.
# Taichi functions must be called from the Taichi-scope.
@ti.func#only be called by Taichi kernels or other Taichi functions.
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])
#Kernel arguments must be type-hinted (if any).
@ti.kernel#perform computation.
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

# GUI system
gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i = i + 1

#循环
# @ti.kernel
# 第一种：range-for loops 可嵌套
# 循环在最外层才能被并行
# def fill():
#     for i in range(10): # Parallelized
#         x[i] += i
#         s = 0
#         for j in range(5): # Serialized in each parallel thread
#             s += j
#         y[i] = s

# @ti.kernel
# def fill_3d():
#     # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0 <= k < 9
#     for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
#         x[i, j, k] = i + j + k

# 第二种：Struct-for loops 用于稀疏计算
# are particularly useful when iterating over (sparse) field elements.
# 例如 for i, j in pixels loops over all the pixel coordinates


# 并行循环中不能 break


## kernel
# 参数 ti.Matrix, or ti.Vector 值传递
# 参数个数上限：8 在 OpenGL 后端，64 在 CPU、Vulkan、CUDA 或 Metal 上
#标量参数中的元素数始终为 1。
#ti.Matrixa或 a中元素的ti.Vector数量是它们内部的实际标量数量。

# kernel最多只有一个返回值 标量、ti.Matrix、 或ti.Vector
# 最多只有一个 return 语句
# At most 30 elements in the return value


## function
# 不能递归
# 支持多个参数， 支持标量、ti.Matrix和ti.Vector作为参数类型
# 不需要键入提示参数，您可以在参数中包含无限数量的元素
# 参数值传递
# 返回值：可以是标量、ti.Matrix、ti.Vector、ti.Struct等
# 可以有多个返回值
# 返回元素数目无限制
# 不能有多个return语句
# 返回值类型可不给出