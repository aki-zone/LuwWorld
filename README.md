# render-LuwWorld

基于CPython加速运算的3D软光栅渲染器

<img alt="zbuffer corrected monkey" src="https://github.com/tvytlx/render-py/raw/master/res/monkey_wireframe.png" alt="monkey" width="420"> <img src="https://github.com/tvytlx/render-py/raw/master/res/monkey_zbuffer.png" alt="wireframe monkey" width="420">


### 已完成：
- [x] 基本渲染管道
- [x] 线框渲染
- [x] z-buffer渲染
- [x] 纹理解析
- [ ] 3D空间视角
- [ ] 切面裁切
- [ ] Pyqt化

### 部署样例:

```
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace && python try.py
```
