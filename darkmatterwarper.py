#!/usr/bin/python


from PIL import Image, ImageDraw
from numpy import linalg as la
from numpy.linalg import norm

from numpy import mod, floor, dot, cross, array, eye, pi, linalg, vstack, arccos, arctan2
from math import cos, sin, sqrt, acos, asin, exp
import sys, getopt


def usage(): 
	print "usage: ", sys.argv[0], " <frame[s]> [<start> [<stop> [<step>]]]"
	print "  <frame[s]>   If only this argument is given, "
	print "               the is the fractional frame number in 0...1."
	print "               If there are more arguments, then frames is "
	print "               total number of frames of the whole animation."
	print "  <start>      Number of the first from to render."
	print "  <stop>       Number of the last from to render up."
	print "  <step>       Number of frames to skip, if stop is specified."
	exit()
	
argc = len(sys.argv)
time = 0
start = 0
stop = 0
step = 1

if argc < 2:
	usage()

if argc >= 2:
	frames = 1
	time = float(sys.argv[1])

if argc >= 3:
	frames = int(sys.argv[1])
	start = int(sys.argv[2])

if argc >= 4:
	stop = int(sys.argv[3])

if argc >= 5:
	step = int(sys.argv[4])


# output options
width = 1280
height = 720
outpath = "/home/kevinh/Videos/darkmatter/frames"


# background options 
wrap = 3
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-enhanced.png"
background = Image.open(filename) 


# view options
povs = array([[15,3,2],[15,-3,-2]])
ups = array([[0,0,1],[0,0,1]])
fov = 0.7


# foreground lensing
Mdm = 0.04
Rdm = 0.25
vdms = array([[15,4,-7],[10,-4,7]])


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
	return array([
		[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 	 	2*(b*d+a*c)],
		[2*(b*c+a*d), 	  a*a+c*c-b*b-d*d,	2*(c*d-a*b)],
		[2*(b*d-a*c), 	  2*(c*d+a*b), 	 	a*a+d*d-b*b-c*c]
	])


# TODO add unlimited control points
def interpolate(frame, M):
	return frame * M[0] + (1 - frame) * M[1]
	

def deflectDarkMatter(ray,vdm,Mdm):
	"returns the deflected ray"
	r = norm(ray)
	b = norm(vdm)
	if r > 0:
		ray /= r
		vdm /= b
		theta = arccos(dot(ray,vdm))
		x = theta / b
	else:
		x = 0
	axis = cross(ray,vdm)
	phi = - Mdm * x / (x**2 + Rdm**2)
	R = rotationMatrix(axis,phi)
	u = dot(R,ray)
	return u


def orthagonalMatrix( k, up ):
	"creates an orthogonal matrix from two vectors"
	x = cross(k,up)
	y = cross(x,k)
	M = vstack([ x/norm(x), y/norm(y), k/norm(k) ])
	return M


def orthographicProjection( (i,j), k, up ):
	"orthogonal projection from a ray, up and pixel values"
	M = orthagonalMatrix(k,up)
	v = M[2] + i*M[0] + j*M[1]
	return v


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


def render(time):
	pov = interpolate(time, povs)
	up = interpolate(time, ups)
	vdm = interpolate(time, vdms)
	image = Image.new("RGB", (width, height))
	for i in range(0, width):
		for j in range(0, height):
			ray = (float(2*i - width) * fov / float(width), 
				   float(2*j - height) * fov / float(width))
			coords = orthographicProjection( ray, pov, up )
			coords = deflectDarkMatter( coords, vdm, Mdm )
			angles = sphereicalProjection( coords )
			pixel = sphericalImageCoordiantes( angles, background.size )
			color = background.getpixel(pixel)
			image.putpixel((i,j), color)
	out = outpath + "/frame" + format(frame,'06') + ".png"
	print "Saving to " + out
	#image.show()
	image.save(out, "PNG")


if start == stop:
	render(time)
else:
	for frame in range(start, stop, step):
		time = float(frame)/float(frames)
		render(time)


# write to stdout
# im.save(sys.stdout, "PNG")
