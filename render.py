from data_types_module import dword, word, char
from utils import color
from obj import Obj
from collections import namedtuple


BLACK = color(0, 0, 0)
WHITE = color(1, 1, 1)
RED = color(1, 0, 0)

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z','w'])


def sum(v0, v1):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element sum
    """
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element substraction
    """
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element multiplication
    """
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
	
def length(v0):
    """
        Input: 1 size 3 vector
        Output: Scalar with the length of the vector
    """  
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    """
        Input: 1 size 3 vector
        Output: Size 3 vector with the normal of the vector
    """  
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def bbox(*vertices):
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]

    xs.sort()
    ys.sort()

    xmin = xs[0]
    xmax = xs[-1]
    ymin = ys[0]
    ymax = ys[-1]

    return xmin, xmax, ymin, ymax

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )

def baryCoords(A, B, C, P):
    # u es para la A, v es para B, w para C
    try:
        u = ( ((B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        v = ( ((C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w

class Render(object):

	#constructor
	def __init__(self):
		self.framebuffer = []
		self.curr_color = WHITE

	def glCreateWindow(self, width, height):
		#width and height for the framebuffer
		self.width = width
		self.height = height

	def glInit(self):
		self.curr_color = BLACK

	def glViewport(self, x, y, width, height):
		self.viewportX = x
		self.viewportY = y
		self.viewportWidth = width
		self.viewportHeight = height

	def glClear(self):
		self.framebuffer = [[BLACK for x in range(
		    self.width)] for y in range(self.height)]
		
		#Zbuffer (buffer de profundidad)
		self.zbuffer = [ [ -float('inf') for x in range(self.width)] for y in range(self.height) ]
		

	def glClearColor(self, r, g, b):
		clearColor = color(
				round(r * 255),
				round(g * 255),
				round(b * 255)
			)

		self.framebuffer = [[clearColor for x in range(
		    self.width)] for y in range(self.height)]

	def glVertex(self, x, y):
		#las funciones fueron obtenidas de https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glViewport.xml
		X = round((x+1) * (self.viewportWidth/2) + self.viewportX)
		Y = round((y+1) * (self.viewportHeight/2) + self.viewportY)
		self.point(X, Y)
		

	def glVertex_coord(self, x, y, color= None):
		if x >= self.width or x < 0 or y >= self.height or y < 0:
			return
		try:
			self.framebuffer[y][x] = color or self.curr_color
		except:
			pass

	def glColor(self, r, g, b):
		self.curr_color = color(round(r * 255), round(g * 255), round(b * 255))

	def point(self, x, y):
		self.framebuffer[x][y] = self.curr_color

	def glFinish(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(char("B"))
		archivo.write(char("M"))
		archivo.write(dword(14+40+self.width*self.height))
		archivo.write(dword(0))
		archivo.write(dword(14+40))

		#Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		#Pixeles, 3 bytes cada uno

		for x in range(self.height):
			for y in range(self.width):
				archivo.write(self.framebuffer[x][y])

		#Close file
		archivo.close()

	#class implementation
	def glLine(self, x0, y0, x1, y1):
		x0 = round(( x0 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		x1 = round(( x1 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		y0 = round(( y0 + 1) * (self.viewportHeight / 2 ) + self.viewportY)
		y1 = round(( y1 + 1) * (self.viewportHeight / 2 ) + self.viewportY)

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5

		m = dy/dx
		y = y0

		for x in range(x0, x1 + 1):
			if steep:
				self.glVertex_coord(y, x)
			else:
				self.glVertex_coord(x, y)

			offset += m
			if offset >= limit:
				y += 1 if y0 < y1 else -1
				limit += 1

	#Homework implementation
	#Reference: http://www.dgp.toronto.edu/~karan/courses/csc418/lectures/BRESENHAM.HTML
	# def glLine(self, x0, y0, x1, y1):
	# 	x0 = round(( x0 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
	# 	x1 = round(( x1 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
	# 	y0 = round(( y0 + 1) * (self.viewportHeight / 2 ) + self.viewportY)
	# 	y1 = round(( y1 + 1) * (self.viewportHeight / 2 ) + self.viewportY)


	# 	dy = abs(y1 - y0)
	# 	dx = abs(x1 - x0)
		
	# 	steep = dy > dx

	# 	if steep:
	# 		x0, y0 = y0, x0
	# 		x1, y1 = y1, x1
		
	# 	if x0 > x1:
	# 		x0, x1 = x1, x0
	# 		y0, y1 = y1, y0
		
	# 	dy = abs(y1 - y0)
	# 	dx = abs(x1 - x0)
		

	# 	y = y0
	# 	offset = 2 *dy - dx

	# 	incre = 2* dy
	# 	decre = 2 * (dy - dx)

	# 	for x in range(x0, x1 + 1):
	# 		offset += 2*dy
	# 		if steep:
	# 			self.glVertex_coord(x, y)
	# 		else:
	# 			self.glVertex_coord(y, x)
	# 		if offset > 0 :
	# 			y += 1
	# 			offset += decre
	# 		else:
	# 			offset += incre
			

		

	def glLine_coord(self, x0, y0, x1, y1): # Window coordinates

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5
		
		try:
			m = dy/dx
		except ZeroDivisionError:
			pass
		else:
			y = y0

			for x in range(x0, x1 + 1):
				if steep:
					self.glVertex_coord(y, x)
				else:
					self.glVertex_coord(x, y)

				offset += m
				if offset >= limit:
					y += 1 if y0 < y1 else -1
					limit += 1

	def transform(self, vertex, translate=V3(0,0,0), scale=V3(1,1,1)):
		return V3(round(vertex[0] * scale.x + translate.x),round(vertex[1] * scale.y + translate.y),round(vertex[2] * scale.z + translate.z))


	def loadModel(self, filename, translate=V3(0,0,0), scale=V3(1,1,1), texture = None, isWireframe = False):
		model = Obj(filename)

		light = V3(0,0,1)

		for face in model.faces:

			vertCount = len(face)

			if isWireframe:
				for vert in range(vertCount):
					
					v0 = model.vertices[ face[vert][0] - 1 ]
					v1 = model.vertices[ face[(vert + 1) % vertCount][0] - 1]

					x0 = round(v0[0] * scale[0]  + translate[0])
					y0 = round(v0[1] * scale[1]  + translate[1])
					x1 = round(v1[0] * scale[0]  + translate[0])
					y1 = round(v1[1] * scale[1]  + translate[1])

					self.glLine_coord(x0, y0, x1, y1)
			else:
				v0 = model.vertices[ face[0][0] - 1 ]
				v1 = model.vertices[ face[1][0] - 1 ]
				v2 = model.vertices[ face[2][0] - 1 ]
				if vertCount > 3:
					v3 = model.vertices[ face[3][0] - 1 ]

				v0 = self.transform(v0,translate, scale)
				v1 = self.transform(v1,translate, scale)
				v2 = self.transform(v2,translate, scale)
				if vertCount > 3:
					v3 = self.transform(v3,translate, scale)

				if texture:
					vt0 = model.texcoords[face[0][1] - 1]
					vt1 = model.texcoords[face[1][1] - 1]
					vt2 = model.texcoords[face[2][1] - 1]
					vt0 = V2(vt0[0], vt0[1])
					vt1 = V2(vt1[0], vt1[1])
					vt2 = V2(vt2[0], vt2[1])
					if vertCount > 3:
						vt3 = model.texcoords[face[3][1] - 1]
						vt3 = V2(vt3[0], vt3[1])
				else:
					vt0 = V2(0,0) 
					vt1 = V2(0,0) 
					vt2 = V2(0,0) 
					vt3 = V2(0,0) 

				normal = cross(sub(V3(v1.x,v1.y,v1.z),V3(v0.x,v0.y,v0.z)), sub(V3(v2.x,v2.y,v2.z),V3(v0.x,v0.y,v0.z)))
				normal = norm(normal)
				intensity = dot(normal, light)

				if intensity >=0:
					self.triangle_bc(v0,v1,v2, texture = texture, texcoords = (vt0,vt1,vt2), intensity = intensity )
					if vertCount > 3: #asumamos que 4, un cuadrado
						self.triangle_bc(v0,v2,v3, texture = texture, texcoords = (vt0,vt2,vt3), intensity = intensity)

	def drawPolygons(self, points):
		
		vertices = len(points)

		for v in range(vertices):
			
			x0 = points[v][0]
			y0 = points[v][1]

			x1 = points[(v +1)% vertices][0]
			y1 = points[(v +1) % vertices][1]

			self.glLine_coord(x0,y0,x1,y1)
			

	#Reference: https://handwiki.org/wiki/Even%E2%80%93odd_rule
	def is_point_in_path(self, x, y, poly):
		num = len(poly)
		i = 0
		j = num - 1
		c = False
		for i in range(num):
			if ((poly[i][1] > y) != (poly[j][1] > y)) and \
					(x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
									(poly[j][1] - poly[i][1])):
				c = not c
			j = i
		return c


	def fillPoly(self, poly, color):
		for x in range(self.width):
			for y in range(self.height):
				if self.is_point_in_path(x,y,poly):
					self.framebuffer[y][x] = color
		
	def glZBuffer(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(bytes('B'.encode('ascii')))
		archivo.write(bytes('M'.encode('ascii')))
		archivo.write(dword(14 + 40 + self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(14 + 40))

		# Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		# Minimo y el maximo
		minZ = float('inf')
		maxZ = -float('inf')
		for x in range(self.height):
			for y in range(self.width):
				if self.zbuffer[x][y] != -float('inf'):
					if self.zbuffer[x][y] < minZ:
						minZ = self.zbuffer[x][y]

					if self.zbuffer[x][y] > maxZ:
						maxZ = self.zbuffer[x][y]

		for x in range(self.height):
			for y in range(self.width):
				depth = self.zbuffer[x][y]
				if depth == -float('inf'):
					depth = minZ
				depth = (depth - minZ) / (maxZ - minZ)
				archivo.write(color(depth,depth,depth))

		archivo.close()


	def triangle(self, A, B, C, color = None):
        
		def flatBottomTriangle(v1,v2,v3):
            #self.drawPoly([v1,v2,v3], color)
			for y in range(v1.y, v3.y + 1):
				xi = round( v1.x + (v3.x - v1.x)/(v3.y - v1.y) * (y - v1.y))
				xf = round( v2.x + (v3.x - v2.x)/(v3.y - v2.y) * (y - v2.y))

				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf + 1):
					self.glVertex_coord(x,y, color or self.curr_color)

		def flatTopTriangle(v1,v2,v3):
			for y in range(v1.y, v3.y + 1):
				xi = round( v2.x + (v2.x - v1.x)/(v2.y - v1.y) * (y - v2.y))
				xf = round( v3.x + (v3.x - v1.x)/(v3.y - v1.y) * (y - v3.y))

				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf + 1):
					self.glVertex_coord(x,y, color or self.curr_color)

        # A.y <= B.y <= Cy
		if A.y > B.y:
			A, B = B, A
		if A.y > C.y:
			A, C = C, A
		if B.y > C.y:
			B, C = C, B

		if A.y == C.y:
			return

		if A.y == B.y: #En caso de la parte de abajo sea plana
			flatBottomTriangle(A,B,C)
		elif B.y == C.y: #En caso de que la parte de arriba sea plana
			flatTopTriangle(A,B,C)
		else: #En cualquier otro caso
			# y - y1 = m * (x - x1)
			# B.y - A.y = (C.y - A.y)/(C.x - A.x) * (D.x - A.x)
			# Resolviendo para D.x
			x4 = A.x + (C.x - A.x)/(C.y - A.y) * (B.y - A.y)
			D = V2( round(x4), B.y)
			flatBottomTriangle(D,B,C)
			flatTopTriangle(A,B,D)

    #Barycentric Coordinates
	def triangle_bc(self, A, B, C, _color = WHITE, texture = None, texcoords = (), intensity = 1):
		#bounding box
		minX = min(A.x, B.x, C.x)
		minY = min(A.y, B.y, C.y)
		maxX = max(A.x, B.x, C.x)
		maxY = max(A.y, B.y, C.y)

		for x in range(minX, maxX + 1):
			for y in range(minY, maxY + 1):
				if x >= self.width or x < 0 or y >= self.height or y < 0:
					continue

				u, v, w = baryCoords(A, B, C, V2(x, y))

				if u >= 0 and v >= 0 and w >= 0:

					z = A.z * u + B.z * v + C.z * w
					if z > self.zbuffer[y][x]:
						
						b, g , r = _color
						b /= 255
						g /= 255
						r /= 255

						b *= intensity
						g *= intensity
						r *= intensity

						if texture:
							ta, tb, tc = texcoords
							tx = ta.x * u + tb.x * v + tc.x * w
							ty = ta.y * u + tb.y * v + tc.y * w

							texColor = texture.getColor(tx, ty)
							b *= texColor[0] / 255
							g *= texColor[1] / 255
							r *= texColor[2] / 255

						self.glVertex_coord(x, y, color(r,g,b))
						self.zbuffer[y][x] = z