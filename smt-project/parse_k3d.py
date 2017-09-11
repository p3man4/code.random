#####################################################
# (C) 2016 Koh Young Research America (KYRA)        #
#                                                   #
# Proprietary and confidential                      #
#####################################################
import struct
import argparse
import os
import numpy as np
import re
import process
import ctypes


class K3DException(Exception):
    pass


def extract_name(filename):
    """
    Used to extract the name from the filename of the component file
    in the form /asd/asd/NAME_21_23.K3d
    Two number suffixes need to be removed and the remnant is treated
    as the component name.
    :param filename:
    :return:
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    pattern = "([0-9a-zA-Z_\-\.]+)_[0-9]+_[0-9]+$"
    g = re.search(pattern, name)
    if g is not None:
        name = g.groups()[0]
    return name


def read_k3d(filename, update_component_id=False):
    """
    Routine to read a K3D file. Returns the dictionary with field from the K3D file
    The 'name' field is extracted from the basename of the filename stripped from extension.
    :param filename: path to a K3D file
    :return: dictionary containing the fields from the file
    """
    f = open(filename, "rb")
    content = f.read()
    f.close()
    if len(content) < 260:
        raise Exception("K3D File %s corrupt" % filename)
    name = None
    if update_component_id:
        name = extract_name(filename)
    return read_k3ds(content, name=name)


def read_k3ds(content, name = None):
    """
    Routine to read a K3D file. Returns the dictionary with field from the K3D file
    The 'name' field is extracted from the basename of the filename stripped from extension.
    :param filename: path to a K3D file
    :return: dictionary containing the fields from the file
    """
    ret_val = {}
    version = struct.unpack("f", content[16:20])
    scale_x = struct.unpack("f", content[20:24])
    scale_y = struct.unpack("f", content[24:28])
    # notused1 = struct.unpack("f", content[28:32])
    # notused2 = struct.unpack("f", content[32:36])
    # notused3 = struct.unpack("h", content[36:38])
    # notused4 = struct.unpack("h", content[38:40])
    inspection_date = struct.unpack("ssssssssss", content[40:50])
    inspection_time = struct.unpack("ssssssss", content[50:58])
    board_name = struct.unpack("ssssssssssssssssssss", content[58:78])
    # notused5 = struct.unpack("h", content[78:80])
    # notused6 = struct.unpack("h", content[80:82])
    # notused7 = struct.unpack("h", content[82:84])
    # notused8 = struct.unpack("h", content[84:86])
    image_width = struct.unpack("i", content[86:90])
    image_height = struct.unpack("i", content[90:94])
    kohyoung_id = struct.unpack("i", content[94:98])
    pad_id = struct.unpack("f", content[98:102])
    pad_type = struct.unpack("h", content[102:104])
    # notused9 = struct.unpack("f", content[104:108])
    result = struct.unpack("h", content[108:110])
    volume_result = struct.unpack("f", content[110:114])
    zmap_height = struct.unpack("f", content[114:118])
    offset_x_result = struct.unpack("f", content[118:122])
    offset_y_result = struct.unpack("f", content[122:126])
    center_x_from_origin = struct.unpack("f", content[126:130])
    center_y_from_origin = struct.unpack("f", content[130:134])
    pad_size_width = struct.unpack("f", content[134:138])
    pad_size_height = struct.unpack("f", content[138:142])
    area_size = struct.unpack("f", content[142:146])
    component_id = struct.unpack("ssssssssssssssssssss", content[146:166])
    pin_number = struct.unpack("sssss", content[166:171])
    panel_id = struct.unpack("h", content[171:173])
    real_volume_result = struct.unpack("f", content[173:177])
    real_area_result = struct.unpack("f", content[177:181])
    pad_spec = struct.unpack("f", content[181:185])
    # notused10 = struct.unpack("h", content[185:187])
    # notused11 = struct.unpack("f", content[187:191])
    # notused12 = struct.unpack("f", content[191:195])
    # notused13 = struct.unpack("f", content[195:199])
    # notused14 = struct.unpack("f", content[199:203])
    # notused15 = struct.unpack("h", content[203:205])
    stencil_height = struct.unpack("f", content[205:209])
    # notused16 = struct.unpack("f", content[209:213])
    # notused17 = struct.unpack("f", content[213:217])
    extend_2d_width = struct.unpack("i", content[217:221])
    extend_2d_height = struct.unpack("i", content[221:225])
    roi_left = struct.unpack("i", content[225:229])
    roi_top = struct.unpack("i", content[229:233])
    roi_width = struct.unpack("i", content[233:237])
    roi_height = struct.unpack("i", content[237:241])
    if name is None:
        ret_val['name'] = ("".join(component_id)).rstrip(' \t\r\n\0')
    else:
        ret_val['name'] = name
    ret_val['version'] = version[0]
    ret_val['scale_x'] = scale_x[0]
    ret_val['scale_y'] = scale_y[0]
    ret_val['inspection_date'] = "".join(inspection_date).rstrip(' \t\r\n\0')
    ret_val['inspection_time'] = "".join(inspection_time).rstrip(' \t\r\n\0')
    ret_val['board_name'] = "".join(board_name)
    ret_val['image_width'] = image_width[0]
    ret_val['image_height'] = image_height[0]
    ret_val['kohyoung_id'] = kohyoung_id[0]
    ret_val['pad_id'] = pad_id[0]
    ret_val['pad_type'] = pad_type[0]
    ret_val['result'] = result[0]
    ret_val['volume_result'] = volume_result[0]
    ret_val['zmap_height'] = zmap_height[0]
    ret_val['offset_x_result'] = offset_x_result[0]
    ret_val['offset_y_result'] = offset_y_result[0]
    ret_val['center_x_from_origin'] = center_x_from_origin[0]
    ret_val['center_y_from_origin'] = center_y_from_origin[0]
    ret_val['pad_size_width'] = pad_size_width[0]
    ret_val['pad_size_height'] = pad_size_height[0]
    ret_val['area_size'] = area_size[0]
    ret_val['component_id'] = "".join(component_id).rstrip(' \t\r\n\0')
    ret_val['pin_number'] = "".join(pin_number).rstrip(' \t\r\n\0')
    ret_val['panel_id'] = panel_id[0]
    ret_val['real_volume_result'] = real_volume_result[0]
    ret_val['real_area_result'] = real_area_result[0]
    ret_val['pad_spec'] = pad_spec[0]
    ret_val['stencil_height'] = stencil_height[0]
    ret_val['extend_2d_width'] = extend_2d_width[0]
    ret_val['extend_2d_height'] = extend_2d_height[0]
    ret_val['roi_left'] = roi_left[0]
    ret_val['roi_top'] = roi_top[0]
    ret_val['roi_width'] = roi_width[0]
    ret_val['roi_height'] = roi_height[0]
    image_2d_size = (image_width[0] * image_height[0])
    if not 241 + image_2d_size < len(content):
        raise K3DException("K3d file invalid, not enough data to parse image")
    image = (np.frombuffer(content[241:241 + image_2d_size], dtype=np.uint8))
    if image_height[0] * image_width[0] < 0:
        raise K3DException("K3d file invalid, image dimensions are negative")
    try:
        image.resize((image_height[0], image_width[0]))
    except:
        raise K3DException("K3d file invalid, cannot resize the image")
    ret_val['img_gray'] = image
    if not 241 + 1 + image_2d_size + 2 * image_2d_size < len(content):
        raise K3DException("K3d file invalid, not enough data to parse image")

    image3d = np.frombuffer(content[241 + 1 + image_2d_size:241 + 1 + image_2d_size + 2 * image_2d_size], dtype=np.int16)
    image3d = image3d.reshape((image_height[0], image_width[0]))
    image3d = image3d.copy().astype(np.float16)
    ret_val['img_3d'] = image3d
    if version[0] > 3.0:
        offset = 241 + 1 + image_2d_size + 2 * image_2d_size + 82
        if not offset + image_2d_size < len(content):
            raise K3DException("K3d file invalid, not enough data to parse image")
        image2dr = np.frombuffer(content[offset:offset + image_2d_size], dtype=np.uint8)
        offset += image_2d_size + 1
        if not offset + image_2d_size < len(content):
            raise K3DException("K3d file invalid, not enough data to parse image")
        image2dg = np.frombuffer(content[offset:offset + image_2d_size], dtype=np.uint8)
        offset += image_2d_size + 1
        if not offset + image_2d_size < len(content):
            raise K3DException("K3d file invalid, not enough data to parse image")
        image2db = np.frombuffer(content[offset:offset + image_2d_size], dtype=np.uint8)
        image2dr.resize((image_height[0], image_width[0]))
        image2dg.resize((image_height[0], image_width[0]))
        image2db.resize((image_height[0], image_width[0]))
        image_BGR = np.dstack((image2db, image2dg, image2dr))
        ret_val['img_bgr'] = image_BGR
    return ret_val


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go through board pictures and try to find electrical components")
    group = parser.add_argument_group()
    group.add_argument("-f", "--file", default="some_file", help="File")
    args = parser.parse_args()
    ret_val = read_k3d(args.file)
    process.pre_process_k3d(ret_val)
