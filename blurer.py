from PIL import Image, ImageFilter
 
filename = "~/Videos/Spiral Galaxy/images/12billionyears-hd.jpg"
original = Image.open(filename) # load an image from the hard drive
blurred = original.filter(ImageFilter.BLUR) # blur the image
  
original.show() # display both images
blurred.show()
