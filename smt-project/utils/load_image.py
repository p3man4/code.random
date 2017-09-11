import skimage.io
import skimage.transform



class image_class(object):

    def __init__(self):
        self.crop_image =None
        self.resized_image = None

    def get_image(self,png_file):
        image = skimage.io.imread(png_file)
        short_edge = min(image.shape[0],image.shape[1])
        yy = int((image.shape[0] - short_edge)/2)
        xx = int((image.shape[1] - short_edge)/2)
        self.crop_image = image[yy:yy+short_edge,xx:xx+short_edge]
        

    def resize_image(self,new_size):
        self.resized_image = skimage.transform.resize(self.crop_image,[new_size,new_size])

