#!/usr/bin/python

from PIL import Image, ImageDraw
from numpy import linalg as la
import numpy as np
 
width = 1280
height = 720
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-hd.jpg"
imageIn = Image.open(filename) # load an image from the hard drive
imageOut = Image.new("RGB", (width, height))

def projectVector( k, u ):
	"projects a vector"
	x = np.cross(k,u)
	y = np.cross(x,k)
	M = np.vstack([ x/la.norm(x), y/la.norm(y), k/la.norm(k) ])
	return M

def sphereicalProjection( omega, k, u ):
	"Projects a vector on to spherical coordinates."
	R = projectVector(k,up)
	M = R[2] + omega[0]*R[0] + omega[1]*R[1]
	x, y, z = M[0], M[1], M[2]
	r = np.sqrt(x**2 + y**2 + z**2)
	phi = np.arctan2(x, y)
	theta = np.arccos(z / r)
	return [phi, theta]

def sphericalImageCoordiantes( phi, theta, eta ):
	v = sphereicalProjection(0.1, k, up)
	return [ np.ceil(v[0] * eta) % eta, np.ceil(v[1] * eta) ]

k = [0,1,0]
up = [0,0,1]
print projectVector(k,up)[1]
print sphereicalProjection([0.1,0.1], k, up)

for i in range(0, width):
	for j in range(0, height):
		imageOut.putpixel((i,j),(255*i*j/width/height,0,0))


# draw = ImageDraw.Draw(im)
# draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)
# del draw

# write to stdout
# im.save(sys.stdout, "PNG")
  
imageOut.show()
