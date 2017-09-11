#####################################################
# (C) 2016 Koh Young Research America (KYRA)        #
#                                                   #
# Proprietary and confidential                      #
#####################################################
# Global imports
import cv2
import numpy as np
# Cherrypy needs to be > 3
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, opening, closing
from skimage.measure import regionprops
# Local imports
import smt_process.CoordTran as coid
import smt_process.common as common
import smt_process.template_2_part as template_2_part

import struct
import ctypes
import re

# Compatibility with opencv2/3
if cv2.__version__.startswith("2"):
    COMP_CORREL = cv2.cv.CV_COMP_CORREL
else:  # if opencv >= 3
    COMP_CORREL = cv2.HISTCMP_CORREL


def findContours(*args, **kwargs):
    """
    Compatibility wrapper for opencv2 version 2.4.X and 3.X
    :param args:
    :param kwargs:
    :return:
    """
    if cv2.__version__.startswith("2"):
        return cv2.findContours(*args, **kwargs)
    else:  # if opencv >= 3
        (im, con, hie) = cv2.findContours(*args, **kwargs)
        return con, hie


def BoxPoints(*args, **kwargs):
    """
    Compatibility wrapper for opencv2 version 2.4.X and 3.X
    :param args:
    :param kwargs:
    :return:
    """
    if cv2.__version__.startswith("2"):
        return cv2.cv.BoxPoints(*args, **kwargs)
    else:  # if opencv >= 3
        return cv2.boxPoints(*args, **kwargs)


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
        region_ret['props'] = region
        region_ret['external_coords'] = tuple(reg_coord)
        peak = 0
        percentile = 95
        while peak == 0:
            peak = np.percentile(depth[tuple(reg_coord)], percentile) % np.max(depth[tuple(reg_coord)])
            percentile -= 5
        region_ret['bodythickness'] = peak
        peak_coords = np.array(
            np.where(depth[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] >= peak * 0.75))
        peak_coords[0] += region.bbox[0]
        peak_coords[1] += region.bbox[1]

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
        contour = cont[np.argmax(areas)]
        contour[:, :, 0] += region.bbox[1]
        contour[:, :, 1] += region.bbox[0]
        rect = cv2.minAreaRect(contour)
        region_ret['internal_contour'] = contour
        region_ret['internal_rect'] = rect
        region_ret['internal_rect_as_cnt'] = np.int0(BoxPoints(rect))
        region_ret['id'] = i
        regions_ret.append(region_ret)
        i += 1
    return regions_ret


def find_nearest_pair(cnt1, cnt2):
    """
    Finds the pair of points where two contours cnt1 and cnt2
    are closes to each other.
    :param cnt1:
    :param cnt2:
    :return:
    """
    nearest1 = -1
    nearest2 = -1
    dist = 1000000000
    for i in xrange(cnt1.shape[0]):
        for j in xrange(cnt2.shape[0]):
            D = ((cnt1[i][0] - cnt2[j][0]) ** 2).sum()
            if D < dist:
                dist = D
                nearest1 = i
                nearest2 = j
    return nearest1, nearest2, dist


def merge_contours(cnt1, cnt2):
    """
    Will merge two opencv contours along the pair of points that
    are nearest to each other.
    :param cnt1:
    :param cnt2:
    :return:
    """
    retval = cnt2
    i, j, _ = find_nearest_pair(cnt1, cnt2)
    for x in range(cnt1.shape[0]):
        index1 = (i + x) % cnt1.shape[0]
        retval = np.insert(retval, j, cnt1[index1], 0)
        j += 1
    return retval


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
                        other_region["external_contour"] = merge_contours(other_region['external_contour'],
                                                                          region["external_contour"])
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


def argsort(seq):
    """
    Returns the order of which when applies to
    the sequence will yield as sorted sequence.
    :param seq:
    :return:
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def align_rects(rect1, rect2):
    """
    Find the arrangement of two rotated rectangles such that they
    match each other.
    :param rect1:
    :param rect2:
    :return:
    """
    angle1 = rect1[2]
    angle2 = rect2[2]
    if np.abs((angle1 - angle2) % 180) < 10:
        return 0
    if 80 < np.abs((angle1 - angle2) % 180) < 100:
        return 1
    return -1


def swap((a, b)):
    """
    Swap a tuple.
    :return:
    """
    return b, a


def is_square((a, b), threshold=0.05):
    """
    Squareness measure.

    :param threshold:
    :return:
    """
    if np.abs(a - b) < threshold * (a + b):
        return True
    return False


def estimate_leads(regions):
    """
    Will estimate the number of leads in 3d data.
    :param regions:
    :return:
    """
    for (i, region) in enumerate(regions):
        ext_c = region['external_contour'].copy()
        x, y, w, h = cv2.boundingRect(ext_c)
        image = np.zeros((h, w), dtype=np.uint8)
        ext_c[:, :, 0] -= x
        ext_c[:, :, 1] -= y
        int_rect = region['internal_rect']
        ext_rect = region['external_rect']
        if align_rects(int_rect, ext_rect) == 1:
            ext_r_prop = swap(ext_rect[1])
        else:
            ext_r_prop = ext_rect[1]

        scaling_factor = 1.2 * min(float(ext_r_prop[0]) / int_rect[1][0], float(ext_r_prop[1]) / int_rect[1][1])
        if is_square(int_rect[1]):
            scaling_factor = (scaling_factor + 2.0) / 3
        scaled_int_r = [int(int_rect[1][0] * scaling_factor), int(int_rect[1][1] * scaling_factor)]
        scaled_int_rect = ((int_rect[0][0] - x, int_rect[0][1] - y),
                           scaled_int_r,
                           int_rect[2])
        scaled_int_rect_as_cnt = np.int0(BoxPoints(scaled_int_rect))
        cv2.drawContours(image, [ext_c], -1, (255), -1)
        image = opening(image, disk(2))
        image = closing(image, disk(2))
        cv2.drawContours(image, [scaled_int_rect_as_cnt], -1, (0), -1)

        cont, hierarchy = findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_filtered = []
        if len(cont) > 0:
            areas = [cv2.contourArea(cnt) for cnt in cont]
            order = argsort(areas)
            cont_sorted = [cont[i] for i in order]
            areas_sorted = [areas[i] for i in order]
            for cnt in enumerate(cont_sorted):
                if areas_sorted[i] > 25:
                    cont_filtered.append(cnt)
        region['lead_map'] = image
        region['lead_cont'] = cont_filtered


def cube_similarity((w1, h1, d1), (w2, h2, d2)):
    """
    Calculates similarity of rectangles.
    0 - very similar
    >0 - less similar
    :return:
    """
    w_1 = max(w1, h1)
    h_1 = min(w1, h1)
    w_2 = max(w2, h2)
    h_2 = min(w2, h2)
    if w1 * w2 * h1 * h2 * d1 * d2 == 0:
        print "Received box dimensions %f %f %f %f %f %f" % (w1, w2, h1, h2, d1, d2)
        return 100000, 0, 0, 0, 0
    r1 = float(w_1) / w_2
    if r1 < 1.0:
        r1 = 1.0 / r1
    r2 = float(h_1) / h_2
    if r2 < 1.0:
        r2 = 1.0 / r2
    # Similarity in aspect ratios
    r3 = float(h_1) / w_1
    r4 = float(h_2) / w_2
    if r3 > 0 and r4 > 0:
        r5 = r3 / r4
        if r5 < 1.0:
            r5 = 1 / r5
    else:
        r5 = 2
    r6 = float(d1) / d2
    if r6 < 1.0:
        r6 = 1.0 / r6
    return (r1 * r2 * r5 * r6) - 1.0, r1, r2, r5, r6


def pre_process_component(k3dict):
    """
    :param k3dict:
    :return:
    """
    CT = coid.CoordinateTransform(fov_x=0, fov_y=0, width=k3dict['image_width'], height=k3dict['image_height'],
                                  scale_x=k3dict['scale_x'],
                                  scale_y=k3dict['scale_y'])  # 1.25 is a magical number, this is likely wrong
    depth_image_norm = k3dict['img_3d']
    depth_image_inter = cv2.resize(depth_image_norm.astype(np.float), dsize=(100, 100), interpolation=cv2.INTER_LANCZOS4)
    depth_image_inter = cv2.GaussianBlur(depth_image_inter, ksize=(3, 3), sigmaX=1)
    thresh = depth_image_inter[50, 50]
    thresh *= 0.5
    markers = (depth_image_norm > thresh).astype(np.uint8) * 255
    markers = binary_fill_holes(markers, disk(3)).astype(np.uint8) * 255
    markers = ndi.label(markers)[0]
    # Find a nonzero marker close to the center with the highest bin count
    frac_dy = markers.shape[0] / 5
    frac_dx = markers.shape[1] / 5
    center_region = markers[markers.shape[0]/2-frac_dy:markers.shape[0]/2+frac_dy, markers.shape[1]/2-frac_dx: markers.shape[1]/2+frac_dx]
    bins = np.bincount(center_region.flatten())
    if bins.shape[0] == 1:
        raise Exception("Could not detect a component in the center!")
    else:
        center_marker = np.argmax(bins[1:])+1
    markers[np.where(markers != center_marker)] = 0
    regions = get_regions(depth_image_norm, markers)
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
        ww, hh = CT.scale(float(w), float(h))  # changed to scale from inv_scale
        d = float(region['bodythickness']) / 1000.0
    region = regions[0]
    isolated_component = common.subimage(k3dict['img_bgr'], (int(cx), int(cy)), region['internal_rect'][2], int(w),
                                         int(h))
    if isolated_component.shape[0] < isolated_component.shape[1]:
        isolated_component = np.rot90(isolated_component)
    isolated_component_normed_size = cv2.resize(isolated_component, dsize=(20, 20), interpolation=cv2.INTER_CUBIC)
    k3dict['isolated_component'] = isolated_component
    k3dict['isolated_component_normed'] = isolated_component_normed_size.flatten().astype(np.float)
    norm1 = np.sqrt(np.dot(k3dict['isolated_component_normed'], k3dict['isolated_component_normed']))
    k3dict['isolated_component_normed_norm'] = norm1

    k3dict['isolated_component_norm_90'] = np.rot90(isolated_component_normed_size, 1).flatten().astype(np.float)
    norm2 = np.sqrt(np.dot(k3dict['isolated_component_norm_90'], k3dict['isolated_component_norm_90']))
    k3dict['isolated_component_norm_90_norm'] = norm2

    k3dict['isolated_component_norm_180'] = np.rot90(isolated_component_normed_size, 2).flatten().astype(np.float)
    norm2 = np.sqrt(np.dot(k3dict['isolated_component_norm_180'], k3dict['isolated_component_norm_180']))
    k3dict['isolated_component_norm_180_norm'] = norm2

    k3dict['isolated_component_norm_270'] = np.rot90(isolated_component_normed_size, 3).flatten().astype(np.float)
    norm2 = np.sqrt(np.dot(k3dict['isolated_component_norm_270'], k3dict['isolated_component_norm_270']))
    k3dict['isolated_component_norm_270_norm'] = norm2
    hsv_roi = cv2.cvtColor(isolated_component, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 255])
    k3dict['hs_histogram'] = hist
    k3dict['observed_w'] = ww
    k3dict['observed_h'] = hh
    k3dict['observed_d'] = d
    if d == 0:
        raise Exception("Depth zero!, bodythickness %f" % region['bodythickness'])



def pre_process_data(components, callback=None):
    """

    :param components:
    :param callback:
    :return:
    """
    print "Components Length : %d" % len (components)
    for (j, component) in enumerate(components.keys()):
        #print "component:",component
        if not 'instances' in components[component].keys():
            #print "skipping"
            continue
        else:
            print "component : ", component, " (", len (components[component]['instances']), ")"
            #print "pre-processing"
            for (i, instance) in enumerate(components[component]['instances']):
                try:
                    pre_process_component(instance)
                except Exception as inst:
                    components[component]['instances'].pop(i)
                    print inst.args[0]
                # if callback is not None:
                    # try:
                        # callback(j, len(components.keys()))
                    # except:
                        # print "Callback failed!"
                        # raise

def merge_components(known_components, components):

    #print "skipping"

    Merged_components = {}
    for (i, component1) in enumerate(known_components.keys()):
        if not 'instances' in known_components[component1].keys():
            #print "skipping"
            Merged_components[component1] = {}
            Merged_components[component1].setdefault('instances', [])
            Merged_components[component1].setdefault('PKG', [])
            Merged_components[component1]['PKG'] = known_components[component1]['PKG']   
            continue
        else:
            Merged_components[component1] = {}
            Merged_components[component1].setdefault('instances', [])
            Merged_components[component1].setdefault('PKG', [])
            Merged_components[component1]['PKG'] = known_components[component1]['PKG']   

            for (k, instance) in enumerate(known_components[component1]['instances']):
                Merged_components[component1]['instances'].append(known_components[component1]['instances'][k])
                
    for (i, component1) in enumerate(components.keys()):
        if component1 in Merged_components:
            #print "skipping"
            if not 'instances' in components[component1].keys():
                continue
            else:
                for (k, instance) in enumerate(components[component1]['instances']):
                    Merged_components[component1]['instances'].append(components[component1]['instances'][k])
        else:
            Merged_components[component1] = {}
            Merged_components[component1].setdefault('instances', [])
            Merged_components[component1].setdefault('PKG', [])
            Merged_components[component1]['PKG'] = components[component1]['PKG']   

            if not 'instances' in components[component1].keys():
                continue
            else:
                for (k, instance) in enumerate(components[component1]['instances']):
                    Merged_components[component1]['instances'].append(components[component1]['instances'][k])

    return Merged_components

def normalize_process_data(components, callback=None):
    """

    :param components:
    :param callback:
    :return:
    """
    normalized_components = {}
    
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    
    for (j, component) in enumerate(components.keys()):
        #print "component:",component
        if not 'instances' in components[component].keys():
            #print "skipping"
            normalized_components[component] = {}
            normalized_components[component].setdefault('instances', [])
            normalized_components[component].setdefault('PKG', [])
            normalized_components[component]['PKG'] = components[component]['PKG']            
            continue
        else:
            print "(Normalized Component: Length) ",component, len(components[component]['instances'])
            
            if len(components[component]['instances']) > 0:
            
                normalized_components[component] = {}
                normalized_components[component].setdefault('instances', [])
                normalized_components[component].setdefault('PKG', [])
                normalized_components[component]['PKG'] = components[component]['PKG']
                                
                Min_Score = 10000
                Min_Idx = -1
                
                for (i, instance_i) in enumerate(components[component]['instances']):
                    Score = 0
                    Count = 0
                    for (k, instance_k) in enumerate(components[component]['instances']):
                        Score_t = Match_Instances(instance_i, instance_k)
                        Score = Score_t + Score
                        Count = Count + 1
                            
                    if Count > 0:
                        Score = Score / Count
                    else:
                        Score = 10000
                        
                    if Min_Score > Score:
                        Min_Score = Score
                        Min_Idx = i
#                        print "(Score : Idx) ", Min_Score, Min_Idx
                
                if Min_Idx != -1:
                    normalized_components[component]['instances'].append(components[component]['instances'][Min_Idx])
#                    print "Selected Component Due to Lowest Score :", components[component]['instances'][Min_Idx]['fname']

                
                Max_Score = -1
                Max_Idx = -1
                
                for (i, instance_i) in enumerate(components[component]['instances']):
                    Score = 0
                    Count = 0
                    Missing_Idx = -1
                    for (c, component_c) in enumerate(components.keys()):
                    
                        if component_c != component:
                        
                            if not 'instances' in components[component_c].keys():
                                #print "skipping"
                                continue
                            else:                    
                                for (k, instance_k) in enumerate(components[component_c]['instances']):
                                    Score_t = Match_Instances(instance_i, instance_k)
                                    Score = Score_t + Score
                                    Count = Count + 1
                                    
                                    if Score_t < Min_Score:
                                        Missing_Idx = 0
                    if Missing_Idx == 0:
                        Idx1 = -1
                        for (f, instance_f) in enumerate(normalized_components[component]['instances']):
                            if instance_f['fname'] == components[component]['instances'][i]['fname']:
                                Idx1 = 0
                        if Idx1 != 0:
                            normalized_components[component]['instances'].append(components[component]['instances'][i])
#                            print "Selected Component Due to Sub Optimal Score :", components[component]['instances'][i]['fname']
                            
                    if Count > 0:
                        Score = Score / Count
                    else:
                        Score = -1
                        
                    if Max_Score < Score:
                        Max_Score = Score
                        Max_Idx = i
#                        print "(Score : Idx) ", Max_Score, Max_Idx
                
                if Max_Idx != -1:
                    Idx1 = -1
                    for (f, instance_f) in enumerate(normalized_components[component]['instances']):
                        if instance_f['fname'] == components[component]['instances'][Max_Idx]['fname']:
                            Idx1 = 0
                    if Idx1 != 0:
                        normalized_components[component]['instances'].append(components[component]['instances'][Max_Idx])
#                        print "Selected Component Due to Most Discrimitive Score :", components[component]['instances'][Max_Idx]['fname']
            else:
                print "Too few component!!! Skip"
            
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    
    for (j, component) in enumerate(normalized_components.keys()):
        if not 'instances' in normalized_components[component].keys():
            #print "skipping"
            continue
        else:
            print "(Normalized Component: Length) ",component, len(normalized_components[component]['instances'])        
            for (i, instance) in enumerate(normalized_components[component]['instances']):
                print "fname: ", normalized_components[component]['instances'][i]['fname']
        
    return normalized_components

def corr_similarity(arg_im1, arg_im2):
    """

    :param arg_im1:
    :param arg_im2:
    :return:
    """
    if np.prod(arg_im1.shape) < np.prod(arg_im2.shape):
        image1 = arg_im1
        image2A = arg_im2
    else:
        image1 = arg_im2
        image2A = arg_im1
    if (image1.shape[0]-image1.shape[1]) * (image2A.shape[0]-image2A.shape[1]) < 0:
        image2A = np.rot90(image2A)

    image21 = cv2.resize(image2A, dsize=image1.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    image2C = np.rot90(image2A, 2)
    image23 = cv2.resize(image2C, dsize=image1.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    norm1 = np.sqrt(np.dot(image1.flatten().astype(np.float), image1.flatten()))
    norm2 = np.sqrt(np.dot(image21.flatten().astype(np.float), image21.flatten()))
    prod = np.dot(image1.flatten().astype(np.float), image21.flatten())
    sim1 = np.arccos(prod/(norm1*norm2))
    norm2 = np.sqrt(np.dot(image23.flatten().astype(np.float), image23.flatten()))
    prod = np.dot(image1.flatten().astype(np.float), image23.flatten())
    sim3 = np.arccos(prod/(norm1*norm2))
    return min(sim1, sim3)


def fast_corr_sim(ar1, norm1, ar2, norm2, ar2f, norm2f, ar2f1, norm2f1, ar2f2, norm2f2):
    """
    Fast computation of the correlation similarity measure.

    :param ar1:
    :param norm1:
    :param ar2:
    :param norm2:
    :param ar2f:
    :param norm2f:
    :param ar2f1:
    :param norm2f1:
    :param ar2f2:
    :param norm2f2:
    :return:
    """
    # prod = np.dot(ar1, ar2)
    # sim0 = np.arccos(min(prod / (norm1 * norm2), 1.0))
    # prod = np.dot(ar1, ar2f)
    # sim1 = np.arccos(min(prod / (norm1 * norm2f), 1.0))
    # prod = np.dot(ar1, ar2f1)
    # sim2 = np.arccos(min(prod / (norm1 * norm2f1), 1.0))
    # prod = np.dot(ar1, ar2f2)
    # sim3 = np.arccos(min(prod / (norm1 * norm2f2), 1.0))
    ds0 = np.sum(cv2.absdiff(ar1, ar2)).astype(np.float)/np.prod(ar1.shape)
    ds1 = np.sum(cv2.absdiff(ar1, ar2f)).astype(np.float)/np.prod(ar1.shape)
    ds2 = np.sum(cv2.absdiff(ar1, ar2f1)).astype(np.float)/np.prod(ar1.shape)
    ds3 = np.sum(cv2.absdiff(ar1, ar2f2)).astype(np.float)/np.prod(ar1.shape)
    return min(ds0, ds1, ds2, ds3)/255.0
    #return min(sim0, sim1, sim2, sim3)


def match(with_leadcnt,template_part_table,sample_k3d, known_components):
    """
    Given a k3d find the best matching component out of the existing known ones
    :param k3d:
    :param known_components:
    :return:
    """
    pre_process_component(sample_k3d)
    most_similar = ""
    similarity = 100000
    alpha = 0.33
    most_similar_debug = {}

    sample_k3d_id = sample_k3d['component_id']
    sample_k3d_component_nm = template_part_table[sample_k3d_id][0]
    sample_k3d_known_component = known_components[sample_k3d_component_nm]
    sample_k3d_pkg= sample_k3d_known_component['PKG']
    sample_k3d_leadcnt = sample_k3d_pkg['LeadCnt']
    #print "sample_k3d:",sample_k3d_component_nm,"leadcnt:", sample_k3d_leadcnt
    
    known_component_leadcnt_final=-1
    for component in known_components.keys():
        if not 'instances' in known_components[component].keys():
            continue
            #print "Skip : ", component
        else:
            #print "component:", component
            known_component = known_components[component]
            known_component_pkg = known_component['PKG']
            known_component_leadcnt = known_component_pkg['LeadCnt']
            #print "known_component:", component,",leadcnt:",known_component_leadcnt

            if with_leadcnt:
                if sample_k3d_leadcnt != known_component_leadcnt:
                    continue

  #          known_component_leadcnt_final = known_component_leadcnt
            for instance in known_components[component]['instances']:
                hist_dist = cv2.compareHist(sample_k3d['hs_histogram'], instance['hs_histogram'], COMP_CORREL)
                box_dist, r0, r1, r2, r3 = cube_similarity((sample_k3d['observed_w'], sample_k3d['observed_h'], sample_k3d['observed_d']),
                                                           (instance['observed_w'], instance['observed_h'], instance['observed_d']))
                corr_dist = fast_corr_sim(sample_k3d['isolated_component_normed'],
                                          sample_k3d['isolated_component_normed_norm'],
                                          instance['isolated_component_normed'],
                                          instance['isolated_component_normed_norm'],
                                          instance['isolated_component_norm_90'],
                                          instance['isolated_component_norm_90_norm'],
                                          instance['isolated_component_norm_180'],
                                          instance['isolated_component_norm_180_norm'],
                                          instance['isolated_component_norm_270'],
                                          instance['isolated_component_norm_270_norm']
                                          )
                final_measure = 0.5 * alpha * (1.0 - hist_dist) + alpha * box_dist + alpha * 0.5 * corr_dist
                if final_measure < similarity:
                    most_similar = component # jun
                    #most_similar = instance['name']
                    similarity = final_measure
                    most_similar_debug['r0'] = "%2.3f" % r0
                    most_similar_debug['r1'] = "%2.3f" % r1
                    most_similar_debug['r2'] = "%2.3f" % r2
                    most_similar_debug['r3'] = "%2.3f" % r3
                    most_similar_debug['box_dist'] = "%2.3f" % box_dist
                    most_similar_debug['hist_dist'] = "%2.3f" % hist_dist
                    most_similar_debug['corr_dist'] = "%2.3f" % corr_dist

                    known_component_leadcnt_final = known_component_leadcnt
                if False:  # For debugging
                    if sample_k3d['name'].startswith(component):
                        print "Compared to %s, final measure %f" % (component, final_measure)
                        print "hist dist", hist_dist
                        print "box_dist", box_dist
                        print "corr_hist", corr_dist


    #print "most_similar:",most_similar
    memo = "(sample_k3d:"+sample_k3d_component_nm+"leadcnt:"+sample_k3d_leadcnt+"),("+most_similar+","+known_component_leadcnt_final+")"
    if sample_k3d_leadcnt != known_component_leadcnt_final:
        print memo
    
    return most_similar, similarity, most_similar_debug

def Match_Instances(instance1, instance2):
    """
    :return:
    """
    alpha = 0.33
    hist_dist = cv2.compareHist(instance1['hs_histogram'], instance2['hs_histogram'], COMP_CORREL)
    box_dist, r0, r1, r2, r3 = cube_similarity((instance1['observed_w'], instance1['observed_h'], instance1['observed_d']),
                                               (instance2['observed_w'], instance2['observed_h'], instance2['observed_d']))
    corr_dist = fast_corr_sim(instance1['isolated_component_normed'],
                              instance1['isolated_component_normed_norm'],
                              instance2['isolated_component_normed'],
                              instance2['isolated_component_normed_norm'],
                              instance2['isolated_component_norm_90'],
                              instance2['isolated_component_norm_90_norm'],
                              instance2['isolated_component_norm_180'],
                              instance2['isolated_component_norm_180_norm'],
                              instance2['isolated_component_norm_270'],
                              instance2['isolated_component_norm_270_norm']
                              )
    final_measure = 0.5 * alpha * (1.0 - hist_dist) + alpha * box_dist + alpha * 0.5 * corr_dist

    return final_measure
    
    
def pack_k3d(k3dict):
    """
    Packs k3d dictionary into a buffer, returns the buffer and the size of the buffer
    :param k3dict:
    :return:
    """
    image_2d_size = np.prod(k3dict['img_gray'].shape)
    image_3d_size = 2 * np.prod(k3dict['img_3d'].shape)
    size = int(241 + image_2d_size + 1 + 2 * image_2d_size + 82 + 3*image_2d_size+3)
    buf = ctypes.create_string_buffer(size)
    struct.pack_into('f', buf, 16, k3dict['version'])
    struct.pack_into('f', buf, 20, k3dict['scale_x'])
    struct.pack_into('f', buf, 24, k3dict['scale_y'])
    buf[40:50] = k3dict['inspection_date'].ljust(10, chr(0))[:]
    buf[50:58] = k3dict['inspection_time'].ljust(8, chr(0))[:]
    buf[58:78] = k3dict['board_name'].ljust(10, chr(0))[:]
    struct.pack_into('i', buf, 86, k3dict['image_width'])
    struct.pack_into('i', buf, 90, k3dict['image_height'])
    struct.pack_into('i', buf, 94, k3dict['kohyoung_id'])
    struct.pack_into('i', buf, 98, k3dict['pad_id'])
    struct.pack_into('h', buf, 102, k3dict['pad_type'])
    struct.pack_into('h', buf, 108, k3dict['result'])
    struct.pack_into('f', buf, 110, k3dict['volume_result'])
    struct.pack_into('f', buf, 114, k3dict['zmap_height'])
    struct.pack_into('f', buf, 118, k3dict['offset_x_result'])
    struct.pack_into('f', buf, 122, k3dict['offset_y_result'])
    struct.pack_into('f', buf, 126, k3dict['center_x_from_origin'])
    struct.pack_into('f', buf, 130, k3dict['center_y_from_origin'])
    struct.pack_into('f', buf, 134, k3dict['pad_size_width'])
    struct.pack_into('f', buf, 138, k3dict['pad_size_height'])
    buf[146:166] = k3dict['component_id'].ljust(20, chr(0))[:]
    buf[166:171] = k3dict['pin_number'].ljust(5, chr(0))[:]
    struct.pack_into('h', buf, 171, k3dict['panel_id'])
    struct.pack_into('f', buf, 173, k3dict['real_volume_result'])
    struct.pack_into('f', buf, 177, k3dict['real_area_result'])
    struct.pack_into('f', buf, 181, k3dict['pad_spec'])
    struct.pack_into('f', buf, 205, k3dict['stencil_height'])
    struct.pack_into('i', buf, 217, k3dict['extend_2d_width'])
    struct.pack_into('i', buf, 221, k3dict['extend_2d_height'])
    struct.pack_into('i', buf, 225, k3dict['roi_left'])
    struct.pack_into('i', buf, 229, k3dict['roi_top'])
    struct.pack_into('i', buf, 233, k3dict['roi_width'])
    struct.pack_into('i', buf, 237, k3dict['roi_height'])
    image2d_buf = np.getbuffer(k3dict['img_gray'])
    buf[241:241+image_2d_size] = image2d_buf[:]
    image3d_buf = np.getbuffer(k3dict['img_3d'].astype(np.int16))
    buf[241 + image_2d_size + 1:241 + image_2d_size + 1 + image_3d_size] = image3d_buf[:]
    offset = 241 + 1 + image_2d_size + 2 * image_2d_size + 82
    imageb = k3dict['img_bgr'][:, :, 0].astype(np.uint8).copy()
    imageg = k3dict['img_bgr'][:, :, 1].astype(np.uint8).copy()
    imager = k3dict['img_bgr'][:, :, 2].astype(np.uint8).copy()
    rbuf = np.getbuffer(imager)
    gbuf = np.getbuffer(imageg)
    bbuf = np.getbuffer(imageb)
    buf[offset:offset+image_2d_size] = rbuf[:]
    offset += image_2d_size + 1
    buf[offset:offset + image_2d_size] = gbuf[:]
    offset += image_2d_size + 1
    buf[offset:offset + image_2d_size] = bbuf[:]
    offset += image_2d_size + 1
    return buf, offset


def write_k3d(filename, k3dict):
    """
    Packs and saves a k3dict to a file.
    :param filename:
    :param k3dict:
    :return:
    """
    f = open(filename, "wb")
    content, size = pack_k3d(k3dict)
    f.write(content)
    f.close()


def evaluate(with_leadcnt, table, test_set, known_components, callback):
    """
    Run the evaluation metric.

    :param with_leadcnt:
    :param table    
    :param test_set:
    :param known_components:
    :return:
    """
    right = 0
    wrong = 0
    uncertain = 0

    template_part_table = template_2_part.build_table(table)
    global result_f
        
    for (i, component) in enumerate(test_set.keys()):
        for instance in test_set[component]['instances']:
            try:
                (most_similar, similarity, debug) = match(with_leadcnt,template_part_table,instance, known_components)
                correct_id = template_part_table[component][0]
                if correct_id == most_similar:
                    right += 1
                else:
                    print instance['fname']
                    print "Actual component %s, found %s with similarity %s" % (correct_id, most_similar, str(similarity))
                    #print debug
                    if similarity > 0.5:
                        wrong += 1
                        cv2.imwrite("wrong/" + correct_id + "_" + str(wrong) + '.jpg', instance['isolated_component'])
                        cv2.imwrite("wrong/" + correct_id + "_" + str(wrong) + '_guessed_' + most_similar + '_.jpg',
                            known_components[most_similar]['instances'][0]['isolated_component'])
                    else:
                        uncertain += 1
                        cv2.imwrite("uncertain/" + correct_id + "_" + str(uncertain) + '.jpg', instance['isolated_component'])
                        cv2.imwrite("uncertain/" + correct_id + "_" + str(uncertain) + '_guessed_' + most_similar + '_.jpg',
                            known_components[most_similar]['instances'][0]['isolated_component'])
            except:
                print "Couldn't Match!!"
        if callback is not None:
            try:
                callback(i, len(test_set.keys()))
            except:
                print "Callback failed!"
                raise

    print "# right %d" % right
    print "# wrong %d" % wrong
    print "# uncertain %d" % uncertain
    print "Aggressive Accuracy %f" % (float(right) / (right + wrong + uncertain))
    print "Relaxed Accuracy %f" % (float(right + uncertain) / (right + wrong + uncertain))

def compress_dict(known_components):
    """
    Compress some of the images in the reference dictionary
    for storage.

    :param known_components:
    :return:
    """
    for component in known_components.keys():
        if not 'instances' in known_components[component].keys():
            continue
        else:
            for instance in known_components[component]['instances']:
                #print "compress_dict :", instance['fname']
                ret, buf = cv2.imencode(".jpg", instance['img_bgr'])
                instance['bgr_jpg'] = buf
                del instance['img_bgr']
                ret, buf = cv2.imencode(".jpg", instance['img_gray'])
                instance['gray_jpg'] = buf
                del instance['img_gray']
                ret, buf = cv2.imencode(".jpg", instance['isolated_component'])
                instance['isolated_jpg'] = buf
                del instance['isolated_component']
                if 'img_3d' in instance:
                    del instance['img_3d']

def uncompress_dict(known_components):
    """
    Revert to the uncompressed version of the data once the
    dictionary is loaded.

    :param known_components:
    :return:
    """
    for component in known_components.keys():
        if not 'instances' in known_components[component]:
            continue
        else:
            for instance in known_components[component]['instances']:
                #print instance['fname']
                buf = cv2.imdecode(instance['bgr_jpg'], cv2.IMREAD_UNCHANGED)
                instance['img_bgr'] = buf
                del instance['bgr_jpg']
                buf = cv2.imdecode(instance['gray_jpg'], cv2.IMREAD_UNCHANGED)
                instance['img_gray'] = buf
                del instance['gray_jpg']
                buf = cv2.imdecode(instance['isolated_jpg'], cv2.IMREAD_UNCHANGED)
                instance['isolated_component'] = buf
                del instance['isolated_jpg']

if __name__ == "__main__":
    print "Run web_app.py for all the functionality"
