import tensorflow as tf
import pandas as pd
import os

def df_to_dataset(data_dir, csv_path, label_encoder, img_size=(64,64), batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)

    # get paths and label names
    image_paths = df["image_path"].apply(lambda p: os.path.join(data_dir, p)).values
    labels_str = df["label"].values

    # convert label names → numeric ids
    labels = label_encoder.transform(labels_str)

    # create simple TF dataset
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(path, label):
        # read image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, label

    # map preprocessing
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
