" script to generate tfrecord data with images and masks "

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from .mask import get_mask

def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """ Returns a float_list from a float / double. """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint. """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecord(source_file, dest_file):
    """Args:
        filenames ([str]): list with diamonds paths
        dest_file (str): file to write tfrecord data """

    with tf.io.TFRecordWriter(dest_file) as writer:
        img = Image.open(source_file + "/Darkfield_EF.jpg")
        img = np.array(img, dtype=np.uint8)
        mask= get_mask(Path(source_file)).transpose(1, 2, 0)

        feature = {
            'image' : _bytes_feature(img.tobytes()),
            'mask'  : _bytes_feature(mask.tobytes())
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example_proto.SerializeToString())
    
def tfrecord2idx(f, idx):
    """Args:
        f ([str]): list with diamonds paths
        idx (str): file to write tfrecord data """
    f = open(sys.argv[1], 'rb')
    idx = open(sys.argv[2], 'w')

    while True:
        current = f.tell()
        try:
            # length
            byte_len = f.read(8)
            if len(byte_len) == 0:
                break
            # crc
            f.read(4)
            proto_len = struct.unpack('q', byte_len)[0]
            # proto
            f.read(proto_len)
            # crc
            f.read(4)
            idx.write(str(current) + ' ' + str(f.tell() - current) + '\n')
        except Exception:
            print("Not a valid TFRecord file")
            break

    f.close()
    idx.close()