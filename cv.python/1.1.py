import sys
sys.path.insert(0,'/home/junwon/Downloads/PIL-1.1.7/PIL')


import Image
from pylab import *
from numpy import *
import sys
from imtools import *


im = array(Image.open('/home/junwon/Pictures/empire.jpg'))
print int(im.min()), int(im.max())


im2 = array(Image.open('/home/junwon/Pictures/empire.jpg').convert('L'),'f')
print int(im2.min()), int(im2.max())

im2,cdf = histeq(im2)


'''
im3 = 255  - im
print int(im3.min()), int(im3.max())


im4 = (100.0/255)*im + 100
print int(im4.min()), int(im4.max())


im5 = 255.0 * (im/255.0)**2
print int(im5.min()), int(im5.max())


im_list = []
im_list.append(im)
im_list.append(im2)
im_list.append(im3)
im_list.append(im4)
im_list.append(im5)

for i  in np.arange(5):
    figure()
    imshow(im_list[i])
show()
'''
#figure()
#imshow(im)
#show()
'''
figure()
gray()
contour(im,origin='image')
axis('equal')

figure()
hist(im.flatten(),128)
show()
'''
