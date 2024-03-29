import os
from keras.utils.image_utils import load_img
 
def resize_images(image_dir):
    """loads, resizes and saves all training images to fixed size
	WARNING: training images will be overriden

	Parameters:
	image_dir(str): directory of images
    """ 
    
    filenames = []

    for subdir, dirs, files in os.walk(image_dir):
        for file in files: 
            if file.endswith(".ppm"):
                filenames.append(os.path.join(subdir, file))

    import PIL
    for i, _file in enumerate(filenames):
        img = load_img(_file)  # this is a PIL image
        img = img.resize((50, 50), PIL.Image.ANTIALIAS)
        img.save(_file)
        if i % 250 == 0:
            print("%d images resized" % i)
        

image_folder = "data/train"
resize_images(image_folder)