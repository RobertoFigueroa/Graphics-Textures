from render import Render, V2, V3

from obj import Obj

from texture import Texture

from utils import color

my_bmp_file = Render()
my_bmp_file.glInit()
my_bmp_file.glCreateWindow(1000,1000)
my_bmp_file.glClear()

t = Texture('./models/heli.bmp')
my_bmp_file.loadModel('./models/heli.obj', V3(500,500,0), V3(5,5,5), t)


my_bmp_file.glFinish('outs.bmp')
