#!/usr/bin/python

from PIL import Image, ImageDraw
from numpy import linalg as la
import numpy as math

# background galaxies 
width = 1280/2
height = 720/2
wrap = 3
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-enhanced.png"


# view options
k = [2,-5,-1]
up = [0,0,1]
fov = 0.8


# foreground lensing
mdm = 100
vdm = [2,-5,-1]
zdm = 0.5


def angleToDarkMatter(v):
	ndm = vdm / math.norm(vdm)
	theta = math.acos(math.dot(v,vdm))
	print rv
	return rv


def projectVector( k, u ):
	"projects a vector"
	x = math.cross(k,u)
	y = math.cross(x,k)
	M = math.vstack([ x/la.norm(x), y/la.norm(y), k/la.norm(k) ])
	return M


def sphereicalProjection( (i,j), k, u ):
	"Projects a vector onto spherical coordinates."
	R = projectVector(k,up)
	M = R[2] + i*R[0] + j*R[1]
	x, y, z = M[0], M[1], M[2]
	r = math.sqrt(x**2 + y**2 + z**2)
	phi = math.arctan2(x, y)
	theta = math.arccos(z / r)
	return (phi, theta)


def sphericalImageCoordiantes( coords, size ):
	v = sphereicalProjection(coords, k, up)
	point = ( math.mod(math.floor(wrap*v[0]*size[0]/2/math.pi), size[0]), 
			  math.mod(math.floor(wrap*v[1]*size[1]/math.pi), size[1]) )
	return point


print projectVector(k,up)[1]
print sphereicalProjection([0.1,0.1], k, up)

imageIn = Image.open(filename) 
imageOut = Image.new("RGB", (width, height))

for i in range(0, width):
	for j in range(0, height):
		point = (float(2*i - width) * fov / float(width), 
				 float(2*j - height) * fov / float(width))
		coords = sphericalImageCoordiantes(point, imageIn.size)
		color = imageIn.getpixel(coords)
		imageOut.putpixel((i,j), color)

imageOut.show()

# write to stdout
# im.save(sys.stdout, "PNG")
  
