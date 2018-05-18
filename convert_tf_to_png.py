import tensorflow as tf
import numpy as np
import argparse
from skimage import io, exposure, img_as_uint, img_as_float


parser = argparse.ArgumentParser(description ='convert tensorflow records to pngs')
parser.add_argument('--tfrecord', type = str, help = 'TFrecord to convert')
parser.add_argument('--out_directory', type = str, help = 'directory to write png to')


args = parser.parse_args()


def reconstruct_image(record_iterator,tfrecord,out_directory):
    snap_size = 100
    count = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['image_raw']
                  .bytes_list
                  .value[0])

        check_img = np.frombuffer(img_string, dtype=np.uint8)
        check_img = check_img.reshape((snap_size, snap_size, 3))
        count += 1
        im = img_as_uint(check_img)
        io.imsave(out_directory+'/'+tfrecord[0:-14]+'_{}_.png'.format(str(count)), im)

if __name__ == '__main__':
    record_iterator = tf.python_io.tf_record_iterator(path=args.tfrecord)
    reconstruct_image(record_iterator,args.tfrecord,args.out_directory)
