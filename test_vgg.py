import tensorflow as tf
import os
from model_vgg import VGG


slim = tf.contrib.slim

TYPE = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
test_datasets = 'datasets/test'

model_path = 'model/vgg'
batch_size = 50



# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    print(filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.cond(
        tf.image.is_jpeg(image_string),
        lambda: tf.image.decode_jpeg(image_string, channels=3),
        lambda: tf.image.decode_png(image_string, channels=3))
    image_gray = tf.image.rgb_to_grayscale(image_decoded)
    image_gray = tf.cast(image_gray, tf.float32) / 255.0

    label = tf.one_hot(label, len(TYPE))
    return image_gray, label

def create_dataset(filenames, labels, batch_size=batch_size, is_shuffle=True, n_repeats=-1, func_map=_parse_function):
    """create dataset for train and validation dataset"""

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
    dataset = dataset.map(func_map)
    if is_shuffle:
        dataset = dataset.shuffle(buffer_size=1000 + 3 * batch_size)
    dataset = dataset.batch(batch_size).repeat(n_repeats)
    return dataset


filenames_t = []
labels_t = []
for index, type in enumerate(TYPE):
    file_list = [os.path.join(test_datasets,  str(index) + '/' + file)
                 for file in os.listdir(os.path.join(test_datasets, str(index)))
                 if file.endswith('jpg')]
    filenames_t += file_list
    num = len(file_list)
    labels_t += [index for i in range(num)]

test_dataset = create_dataset(filenames_t, labels_t, is_shuffle=False, n_repeats=1)
iterator = test_dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

train_mode = tf.placeholder(tf.bool)

keep_prob = tf.placeholder(tf.float32, [])

model = VGG()

logits = model.predict(input=images, num_classes=len(TYPE), dropout_prob=keep_prob, training=train_mode)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess, open('test.log', 'w') as log:

    if tf.train.latest_checkpoint(model_path) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    else:
        assert 'can not find checkpoint folder path!'

    total_acc = .0
    cnt = 0
    while True:
        try:
            acc_ = sess.run(accuracy, feed_dict={keep_prob: 0.0, train_mode: False})
            total_acc += acc_
            cnt += 1
            log.write("accuracy:{:.2f}\n".format(acc_))
            print("accuracy:{:.2f}\n".format(acc_))
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
            total_acc /= cnt
            log.write("avg accuracy:{:.2f}\n".format(total_acc))
            print("avg accuracy:{:.2f}\n".format(total_acc))
            break

