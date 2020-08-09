'''
Texture class
Reads an bmp file
'''

import struct
from utils import color


class Texture(object):
    def __init__(self, path):
        self.path = path #path of the bmp file
        self.read()
        
    def read(self): #reads the bmp file
        image = open(self.path, 'rb') #read binary mode
        image.seek(10)
        headerSize = struct.unpack('=l', image.read(4))[0]

        image.seek(14 + 4)
        self.width = struct.unpack('=l', image.read(4))[0]
        self.height = struct.unpack('=l', image.read(4))[0]
        image.seek(headerSize)

        self.pixels = []

        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                try:
                    b = ord(image.read(1)) / 255
                    g = ord(image.read(1)) / 255
                    r = ord(image.read(1)) / 255
                    self.pixels[y].append(color(r,g,b))
                except TypeError:
                    continue

        image.close()

    def getColor(self, tx, ty):
        if tx >= 0 and tx <= 1 and ty >= 0 and ty <= 1:
            x = int(tx * self.width)
            y = int(ty * self.height)

            return self.pixels[y][x]
        else:
            return color(0,0,0)