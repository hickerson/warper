from PIL import Image, ImageDraw
 
filename = "/home/kevinh/Videos/Spiral Galaxy/images/12billionyears-hd.jpg"
original = Image.open(filename) # load an image from the hard drive
im = original

draw = ImageDraw.Draw(im)
draw.line((0, 0) + im.size, fill=128)
draw.line((0, im.size[1], im.size[0], 0), fill=128)
del draw

# write to stdout
# im.save(sys.stdout, "PNG")
  
im.show()
