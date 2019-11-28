import sys
import argparse
import os
#from yolo import YOLO, detect_video
from yolo_labeling import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        file_path = input('Input filepath:')
        img_list = os.listdir(file_path)
        if file_path[-1] != "/":
            file_path += "/"

        if os.path.exists(file_path+"result") != True:
           os.mkdir(file_path+"result")
        if os.path.exists(file_path+"Annotations") != True:
           annotation_path = file_path+"Annotations"
           os.mkdir(annotation_path)

        for i in range(len(img_list)):
            img_name = img_list[i]
            print(img_name)

            img_path = file_path + img_name

            try:
                image = Image.open(img_path)
            except:
                print('Open Error! Try again!')
                continue

            else:
                #r_image = yolo.detect_image(image)
                #r_image.show()
                r_image = yolo.detect_image(image,img_name,img_path,annotation_path)
                r_image.save(file_path+"result/"+img_name)

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )


    FLAGS = parser.parse_args()

    print("Image detection mode")
    detect_img(YOLO(**vars(FLAGS)))
