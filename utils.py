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

class_labels = {
    0: "astilbe",
    1: "black_eyed_susan",
    2: "bluebell",
    3: "buttercup",
    4: "calendula",
    5: "carnation",
    6: "colts_foot",
    7: "cowslip",
    8: "crocus",
    9: "daffodil",
    10: "daisy",
    11: "dandelion",
    12: "fritillary",
    13: "iris",
    14: "lily_valley",
    15: "magnolia",
    16: "pansy",
    17: "rose",
    18: "snowdrop",
    19: "sunflower",
    20: "tigerlily",
    21: "tulip",
    22: "water_lily",
    23: "windflower"
};
