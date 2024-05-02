import cv2, os
def load_datasets(folder_path, labels, fixed_size):
    i = 0
    label_set = []
    image_set = []
    for label in labels:
        # join the training data path and each species training folder
        dir = os.path.join(folder_path, label)

        # get the current training label
        current_label = i
        list_images = os.listdir(dir)
        for image in list_images:
            file = os.path.join(dir, image)
            # read the image and resize it to a fixed-size
            cv2_image = cv2.imread(file)    
            cv2_image = cv2.resize(cv2_image, fixed_size)

            # Lưu trữ lại
            label_set.append(current_label)
            image_set.append(cv2_image)

        # Thông báo đã xử lí xong 1 folder
        print("[STATUS] processed folder: {}".format(current_label))
        i += 1

    print("Size of an image: {}", image_set[0].shape)
    return image_set, label_set