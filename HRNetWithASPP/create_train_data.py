import cv2, os, shutil
import numpy as np

if __name__ == '__main__':
    data_path = '/media/feng/新加卷/human/data'
    mask_path = '/media/feng/新加卷/human/result/final'
    train_mask_path = 'data/my_data/TrainVal_parsing_annotations/train_segmentations'
    train_reversed_mask_path = 'data/my_data/TrainVal_parsing_annotations/train_segmentations_reversed'
    val_mask_path = 'data/my_data/TrainVal_parsing_annotations/val_segmentations'
    train_images_path = 'data/my_data/TrainVal_images/train_images'
    val_images_path = 'data/my_data/TrainVal_images/val_images'
    for path in [
            train_mask_path, train_reversed_mask_path, val_mask_path,
            train_images_path, val_images_path
    ]:
        if not os.path.exists(path):
            os.makedirs(path)
    for dir_name in os.listdir(data_path):
        #测试集
        if dir_name in {'20200115', '20200116', '20200117'}:
            for dir2_name in os.listdir(os.path.join(data_path, dir_name)):
                for image_name in os.listdir(
                        os.path.join(data_path, dir_name, dir2_name)):
                    rgb_image_path = os.path.join(data_path, dir_name,
                                                  dir2_name, image_name)
                    shutil.copyfile(
                        rgb_image_path,
                        os.path.join(
                            val_images_path, dir2_name + '_' +
                            image_name.replace('.JPG', '.png')))
                    mask_image_path = os.path.join(mask_path, dir_name,
                                                   dir2_name,
                                                   dir2_name + '_MASK',
                                                   image_name)
                    img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
                    img[img != 0] = 1
                    cv2.imwrite(
                        os.path.join(
                            val_mask_path, dir2_name + '_' +
                            image_name.replace('.JPG', '.png')), img)
                    print('copy', dir2_name + '_' + image_name)
        #训练集
        elif dir_name in {
                '20200102', '20200103', '20200104', '20200105', '20200106',
                '20200107', '20200108', '20200109', '20200110', '20200111',
                '20200112', '20200113', '20200114'
        }:
            for dir2_name in os.listdir(os.path.join(data_path, dir_name)):
                for image_name in os.listdir(
                        os.path.join(data_path, dir_name, dir2_name)):
                    rgb_image_path = os.path.join(data_path, dir_name,
                                                  dir2_name, image_name)
                    shutil.copyfile(
                        rgb_image_path,
                        os.path.join(
                            train_images_path, dir2_name + '_' +
                            image_name.replace('.JPG', '.png')))
                    mask_image_path = os.path.join(mask_path, dir_name,
                                                   dir2_name,
                                                   dir2_name + '_MASK',
                                                   image_name)
                    img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
                    img[img != 0] = 1
                    cv2.imwrite(
                        os.path.join(
                            train_mask_path, dir2_name + '_' +
                            image_name.replace('.JPG', '.png')), img)
                    print('copy', dir2_name + '_' + image_name)
