#!/usr/bin/python

from PIL import Image, ImageDraw
from numpy import linalg as la
import numpy as math
 
width = 1280
height = 720
fov = math.pi
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-hd.jpg"
imageIn = Image.open(filename) # load an image from the hard drive
imageOut = Image.new("RGB", (width, height))

def projectVector( k, u ):
	"projects a vector"
	x = math.cross(k,u)
	y = math.cross(x,k)
	M = math.vstack([ x/la.norm(x), y/la.norm(y), k/la.norm(k) ])
	return M

def sphereicalProjection( (i,j), k, u ):
	"Projects a vector on to spherical coordinates."
	R = projectVector(k,up)
	M = R[2] + i*R[0] + j*R[1]
	x, y, z = M[0], M[1], M[2]
	r = math.sqrt(x**2 + y**2 + z**2)
	phi = math.arctan2(x, y)
	theta = math.arccos(z / r)
	return (phi, theta)

def sphericalImageCoordiantes( coords, eta ):
	v = sphereicalProjection(coords, k, up)
	return ( math.mod(math.ceil(v[0] * eta), eta), 
			 math.mod(math.ceil(v[1] * eta), eta) )

k = [0,1,0]
up = [0,0,1]
print projectVector(k,up)[1]
print sphereicalProjection([0.1,0.1], k, up)

for i in range(0, width):
	for j in range(0, height):
		point = (float(2*i - width) * fov / float(width), 
				 float(2*j - height) * fov / float(width))
		coords = sphericalImageCoordiantes(point, width)
		# print point
		# print coords
		color = imageIn.getpixel(coords)
		imageOut.putpixel((i,j), color)

imageOut.show()

# write to stdout
# im.save(sys.stdout, "PNG")
  
