from LRender import render, Model

# you can export the *.obj format model data from blender
render(
    Model("Res/axe.obj", texture_filename="Res/axe.tga"),
    height=4000,
    width=4000,
    filename="axe.png",
)