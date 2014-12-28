#!/usr/bin/python

from PIL import Image, ImageDraw
from numpy import linalg as la
from numpy.linalg import norm

from numpy import mod, floor, dot, cross, array, eye, pi, linalg, vstack, arccos, arctan2
from math import cos, sin, sqrt, acos, asin

"""
#import numpy.random as nr
#from scipy import weave

def rotation_matrix_weave(axis, theta, mat = None):
	if mat == None:
		mat = eye(3,3)

	support = "#include <math.h>"
	code = 
		double x = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
		double a = cos(theta / 2.0);
		double b = -(axis[0] / x) * sin(theta / 2.0);
		double c = -(axis[1] / x) * sin(theta / 2.0);
		double d = -(axis[2] / x) * sin(theta / 2.0);

		mat[0] = a*a + b*b - c*c - d*d;
		mat[1] = 2 * (b*c - a*d);
		mat[2] = 2 * (b*d + a*c);

		mat[3*1 + 0] = 2*(b*c+a*d);
		mat[3*1 + 1] = a*a+c*c-b*b-d*d;
		mat[3*1 + 2] = 2*(c*d-a*b);

		mat[3*2 + 0] = 2*(b*d-a*c);
		mat[3*2 + 1] = 2*(c*d+a*b);
		mat[3*2 + 2] = a*a+d*d-b*b-c*c;

	weave.inline(code, ['axis', 'theta', 'mat'], support_code = support, libraries = ['m'])

	return mat
"""


def rotationMatrix(axis, theta):
	"rotate about axis by angle theta"
	mat = eye(3,3)
	n = norm(axis)
	if n == 0:
		theta = 0
	else:
		axis = axis / n
	a = cos(theta/2.)
	b, c, d = -axis*sin(theta/2.)
	#print a ,b, c, d
	return array([
		[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 	 	2*(b*d+a*c)],
		[2*(b*c+a*d), 	  a*a+c*c-b*b-d*d,	2*(c*d-a*b)],
		[2*(b*d-a*c), 	  2*(c*d+a*b), 	 	a*a+d*d-b*b-c*c]
	])


# background galaxies 
width = 1280/2
height = 720/2
wrap = 3
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-enhanced.png"


# view options
pov = array([2,-5,-1])
up = array([0,0,1])
fov = 0.8


# foreground lensing
mdm = 0.1
vdm = array([2,-6,-1])
zdm = 0.5
ndm = vdm / norm(vdm)


def deflectDarkMatter(ray,vdm):
	"returns the deflected ray"
	r = norm(ray)
	d = norm(vdm)
	if r > 0:
		ray /= r
		vdm /= d
		theta = arccos(dot(ray,vdm))
	else:
		theta = 0
	axis = cross(ray,ndm)
	phi = - mdm / d / theta
	R = rotationMatrix(axis,phi)
	u = dot(R,ray)
	return u


def projectionMatrix( k, u ):
	"projects a vector"
	x = cross(k,u)
	y = cross(x,k)
	M = vstack([ x/norm(x), y/norm(y), k/norm(k) ])
	return M


def orthographicProjection( (i,j), k, u ):
	M = projectionMatrix(k,up)
	v = M[2] + i*M[0] + j*M[1]
	return v
	#M = R[2] + i*R[0] + j*R[1]
	#return array([M[0], M[1], M[2]])


def sphereicalProjection( u ):
	"Projects a vector onto spherical coordinates."
	r = norm(u)
	phi = arctan2(u[0], u[1])
	theta = arccos(u[2] / r)
	return (phi, theta)


def sphericalImageCoordiantes( angles, size ):
	pixel = ( mod(floor(wrap*angles[0]*size[0]/2/pi), size[0]), 
			  mod(floor(wrap*angles[1]*size[1]/pi), size[1]) )
	return pixel


print projectionMatrix(pov,up)[1]
#print sphereicalProjection([0.1,0.1], pov, up)
#print deflectDarkMatter(pov,pov,ndm)


imageIn = Image.open(filename) 
imageOut = Image.new("RGB", (width, height))


for i in range(0, width):
	for j in range(0, height):
		ray = (float(2*i - width) * fov / float(width), 
			   float(2*j - height) * fov / float(width))
		coords = orthographicProjection( ray, pov, up )
		coords = deflectDarkMatter( coords, ndm )
		angles = sphereicalProjection( coords )
		pixel = sphericalImageCoordiantes( angles, imageIn.size )
		color = imageIn.getpixel(pixel)
		imageOut.putpixel((i,j), color)


imageOut.show()

# write to stdout
# im.save(sys.stdout, "PNG")
