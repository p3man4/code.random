#
# input param: input directory
#
#
import parse_k3d
import os
import cv2
import skimage.io

import skimage.transform
from skimage.measure import regionprops
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import CoordTran as coid
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk,opening,closing
from scipy import ndimage as ndi

k3d_path='/home/junwon/smt-data/Train_0818/Measure_JOB1/C469.k3d'
temp_path='/home/junwon/smt-data/Train_0818/Measure_JOB1/'

def main():
    k3d = parse_k3d.read_k3d(k3d_path)
    print "image_width:", k3d['scale_x']
    img = k3d['img_3d'].astype(int)
    print "image shape:",img.shape

    #cv2.imwrite(temp_path + "temp.jpg",img)
    #img2 = mpimg.imread(temp_path + "temp.jpg")

    #plt.imshow(img)
    #plt.show()
    pre_process_component(k3d)


def findContours(*args, **kwargs):
    if cv2.__version__.startswith("2"):
        return cv2.findContours(*args, **kwargs)
    else:
        (im, con, hie) = cv2.findContours(*args, **kwargs)
        return con, hie



def clean_regions(regions):
    """
    Removes and merges the contours which are close to the
     external bounding box of the component region.
    :param regions:
    :return:
    """
    removed_regions = []
    marked_for_removal = []
    for (i, region) in enumerate(regions):
        center = region['internal_rect'][0]
        for other_region in regions:
            if other_region['props'].area > region['props'].area:
                dist = cv2.pointPolygonTest(other_region['external_rect_as_cnt'], center, True)
                if dist > -0.2 * np.max(other_region['external_rect'][1]):
                    _, _, dist2 = find_nearest_pair(region['external_contour'], other_region['external_contour'])
                    if dist >= 0 or np.sqrt(dist2) < 15:
                        other_region["external_contour"] = merge_contours(other_region['external_contour'], region["external_contour"])
                        rect = cv2.minAreaRect(other_region["external_contour"])

                        other_region['external_rect'] = rect
                        other_region['external_rect_as_cnt'] = np.int0(BoxPoints(rect))
                        marked_for_removal.append(i)
                        break
    if len(marked_for_removal) > 0:
        marked_for_removal.reverse()
        for i in marked_for_removal:
            removed_regions.append(regions.pop(i))
            print "Removing a region"
    return regions, removed_regions


def BoxPoints(*args, **kwargs):
    if cv2.__version__.startswith("2"):
        return cv2.cv.BoxPoints(*args, **kwargs)
    else:
        return cv2.boxPoints(*args, **kwargs)

def pre_process_component(k3dict):
    """
    :param k3dict:
    :return:
    """
    CT = coid.CoordinateTransform(fov_x=0, fov_y=0, width=k3dict['image_width'], height=k3dict['image_height'],
                                  scale_x=k3dict['scale_x'],
                                  scale_y=k3dict['scale_y'])  # 1.25 is a magical number, this is likely wrong
    depth_image_norm = k3dict['img_3d']
    #plt.imshow(depth_image_norm)
    #plt.show()
    depth_image_inter = cv2.resize(depth_image_norm.astype(np.float), dsize=(100, 100), interpolation=cv2.INTER_LANCZOS4)
    depth_image_inter = cv2.GaussianBlur(depth_image_inter, ksize=(3, 3), sigmaX=1)
    #plt.imshow(depth_image_inter)
    #plt.show()
    print "max depth image inter:",np.max(depth_image_inter)
    print "min depth image inter:",np.min(depth_image_inter)
    print "max depth image norm:", np.max(depth_image_norm)
    print "min depth image norm:",np.min(depth_image_norm)
    thresh = depth_image_inter[50, 50]
    print "thresh:",thresh
    thresh *= 0.5
    print "post-thresh:",thresh
    markers = (depth_image_norm > thresh).astype(np.uint8) * 255
    print "min markers:",np.min(markers)
    print "max markers:",np.max(markers)
    markers = binary_fill_holes(markers, disk(3)).astype(np.uint8) * 255
    #plt.imshow(markers)
    #plt.show()
    
    markers = ndi.label(markers)[0]
    print "num_features:",ndi.label(markers)[1]
    print "min markers:",np.min(markers)
    print "max markers:",np.max(markers)
    
    
    #plt.imshow(markers)
    #plt.show()
    print "markers.shape",markers.shape
    # Find a nonzero marker close to the center with the highest bin count
    frac_dy = markers.shape[0] / 5
    frac_dx = markers.shape[1] / 5
    print "frac_dy:",frac_dy
    print "frac_dx:",frac_dx
    a1 = markers.shape[0]/2-frac_dy
    a2 = markers.shape[0]/2+frac_dy
    b1 = markers.shape[1]/2-frac_dx
    b2 = markers.shape[1]/2+frac_dx
    print "a1,a2,b1,b2:",a1,",",a2,",",b1,",",b2
    #center_region = markers[markers.shape[0]/2-frac_dy:markers.shape[0]/2+frac_dy, markers.shape[1]/2-frac_dx: markers.shape[1]/2+frac_dx]
    center_region=markers[a1:a2,b1:b2]
    print "center_region:",center_region 
    
    #plt.imshow(center_region)
    #plt.show()
    bins = np.bincount(center_region.flatten())
    print "bins:",bins.shape[0]
    if bins.shape[0] == 1:
        raise Exception("Could not detect a component in the center!")
    else:
        center_marker = np.argmax(bins[1:])+1
    print "center_marker:",center_marker
    markers[np.where(markers != center_marker)] = 0
    #print "markers:",markers
    #for i in np.arange(len(markers)):
    #    print i,":",markers[i]
    #plt.imshow(markers)
    #plt.show()
    
    regions = get_regions(depth_image_norm, markers)
    print "regions::",regions
    regions, removed_regions = clean_regions(regions)
    regions, removed_regions1 = clean_regions(regions)
    if len(regions) == 0:
        raise Exception("Could not identify the component" + k3dict['component_id'])

    removed_regions.extend(removed_regions1)
    estimate_leads(regions)
    fused = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    w, h, d = (0, 0, 0)
    cx, cy = 0, 0
    ww, hh = 0, 0
    for region in regions:
        cv2.drawContours(fused, [region['external_contour']], -1, (255, 255, 0), 3)
        cv2.drawContours(fused, [region['internal_contour']], -1, (0, 255, 0), 3)
        cv2.drawContours(fused, [region['external_rect_as_cnt']], -1, (255, 255, 0), 2)
        cv2.drawContours(fused, [region['internal_rect_as_cnt']], -1, (0, 255, 0), 2)
        cx, cy = region['internal_rect'][0]
        w, h = region['internal_rect'][1]
        ww, hh = CT.scale(float(w),float(h)) # changed to scale from inv_scale
        d = float(region['bodythickness']) / 1000.0
    region = regions[0]
    isolated_component = common.subimage(k3dict['img_bgr'],(int(cx),int(cy)), region['internal_rect'][2],int(w),int(h))

    k3dict['isolated_component'] = isolated_componene
	

def get_regions(depth, markers):
    """
    Extracts the ROI's based on the depth data and given markers.
    :param depth:
    :param markers:
    :return:
    """
    regions_ret = []
    regions = regionprops(markers)
    # Remove regions whose convex hull overlaps by more than 3%
    # with other regions


    i = 0
    for (i, region) in enumerate(regions):
        if region.area < 25:
            continue
        region_ret = {}
        reg_coord = region.coords.T
        print "reg_coord:",reg_coord
        print "tuple(reg_coord):",tuple(reg_coord)
        #plt.imshow(region)
        #plt.show()

        region_ret['props'] = region
        region_ret['external_coords'] = tuple(reg_coord)

        peak = 0
        percentile = 95

        while peak == 0:
            print "inner peak:",peak
            peak = np.percentile(depth[tuple(reg_coord)], percentile) % np.max(depth[tuple(reg_coord)])
            percentile -= 5
        region_ret['bodythickness'] = peak
        peak_coords = np.array(
            np.where(depth[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] >= peak * 0.75))
        print "peak_coords:",peak_coords
        
        peak_coords[0] += region.bbox[0]
        peak_coords[1] += region.bbox[1]
        
        print "region.bbox[0]:",region.bbox[0]
        print "region.bbox[1]:", region.bbox[1]
        print "peak_coords[0]:", peak_coords[0]
        print "peak_coords[1]:", peak_coords[1]
        print "regions.coords[:,0]:",region.coords[:,0]
        print "regions.coords[:,1]:",region.coords[:,1]
        print "shift <<:", region.coords[:, 0] << 16
        print "shift2 <<:", peak_coords.T[:, 0] << 16

        reg_coord_hash = (region.coords[:, 0] << 16) + region.coords[:, 1]
        peak_coord_hash = (peak_coords.T[:, 0] << 16) + peak_coords.T[:, 1]

        i0 = np.in1d(reg_coord_hash, peak_coord_hash)
        internal_coords = region.coords[i0, :].T
        region_ret['internal_coords'] = tuple(internal_coords)
        bin_im = region.filled_image.astype(np.uint8)
        cont, hierarchy = findContours(bin_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cont) < 1:
            continue
        areas = [cv2.contourArea(cnt) for cnt in cont]
        contour = cont[np.argmax(areas)]
        contour[:, :, 0] += region.bbox[1]
        contour[:, :, 1] += region.bbox[0]
        rect = cv2.minAreaRect(contour)
        region_ret['external_contour'] = contour
        region_ret['external_rect'] = rect
        region_ret['external_rect_as_cnt'] = np.int0(BoxPoints(rect))
        box = np.zeros_like(bin_im)
        box[internal_coords[0] - region.bbox[0], internal_coords[1] - region.bbox[1]] = 1
        cont, hierarchy = findContours(box, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cont) < 1:
            continue
        areas = [cv2.contourArea(cnt) for cnt in cont]

main()



