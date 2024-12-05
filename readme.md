# LuwWorld-py

基于CPython加速运算的3D软光栅渲染器。
<div style="display: flex; justify-content: space-around;">
    <img alt="zbuffer corrected monkey" src="./res/monkey_zbuffer.png" alt="monkey" width="420"> <img src="./res/monkey_wireframe.png" alt="wireframe monkey" width="420">
</div>



### 实验进度：
- [x] 基本渲染管道
- [x] 线框渲染
- [x] z-buffer渲染
- [x] 纹理解析
- [ ] 3D空间视角
- [ ] 切面裁切
- [ ] Pyqt/Numpy空间算子加速

### 部署样例:

```
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace && python try.py
```

### 许可证
在 MIT协议 许可下可用。查看[LICENSE](LICENSE)文件了解更多信息。
