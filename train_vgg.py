import tensorflow as tf
import os
from model_vgg import VGG
import numpy as np


TYPE = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
train_datasets = 'datasets/train'
validation_datasets = 'datasets/val'
learning_rate = 1e-4
decay_steps = 5000
decay_rate = 0.9
summary_path = 'summary'
epoch = 50
steps = 1000
model_path = 'model/vgg'
batch_size = 200

if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(summary_path):
    os.mkdir(summary_path)


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



# train data
filenames_t = []
labels_t = []
for index, type in enumerate(TYPE):
    file_list = [os.path.join(train_datasets,  str(index) + '/' + file)
                 for file in os.listdir(os.path.join(train_datasets, str(index)))
                 if file.endswith('jpg')]
    filenames_t += file_list
    num = len(file_list)
    labels_t += [index for i in range(num)]

randnum = np.random.randint(0, 100)
np.random.seed(randnum)
np.random.shuffle(filenames_t)
np.random.seed(randnum)
np.random.shuffle(labels_t)

train_dataset = create_dataset(filenames_t, labels_t)

# validation data
filenames_v = []
labels_v = []
for index, type in enumerate(TYPE):
    file_list = [os.path.join(validation_datasets, str(index) + '/' + file)
                 for file in os.listdir(os.path.join(validation_datasets, str(index)))
                 if file.endswith('jpg')]
    filenames_v += file_list
    num = len(file_list)
    labels_v += [index for i in range(num)]


randnum = np.random.randint(0, 100)
np.random.seed(randnum)
np.random.shuffle(filenames_v)
np.random.seed(randnum)
np.random.shuffle(labels_v)
val_dataset = create_dataset(filenames_v, labels_v)

# 创建一个feedable iterator
handle = tf.placeholder(tf.string, [])
train_mode = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32, [])

feed_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                  train_dataset.output_shapes)
images, labels = feed_iterator.get_next()

# 创建不同的iterator
train_iterator = train_dataset.make_one_shot_iterator()
val_iterator = val_dataset.make_initializable_iterator()

model = VGG()

logits = model.predict(input=images, num_classes=len(TYPE), dropout_prob=keep_prob, training=train_mode)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)

lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(lr, name='optimizer').minimize(loss, global_step=global_step)

for v in tf.all_variables():
    print(v.name)
    if 'batch_normalization' in v.name:
        tf.summary.histogram(v.name, v)
tf.summary.scalar('learn_rate', lr)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(summary_path)

saver = tf.train.Saver(max_to_keep=5)

init_op = tf.global_variables_initializer()

with tf.Session() as sess, open('train.log', 'w') as log:
    # 生成对应的handle
    sess.run(init_op)
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())

    # 训练
    for n in range(epoch):
        for i in range(steps):

            g_step_, loss_, acc_, train_summary, _ = sess.run([global_step, loss, accuracy, merged, optimizer],
                                                              feed_dict={handle: train_handle, keep_prob: 0.5, train_mode: True})

            print("step:{} loss:{:.2f}, accuracy:{:.2f}".format(g_step_, loss_, acc_))
            log.write("step:{} loss:{:.2f}, accuracy:{:.2f}".format(g_step_, loss_, acc_))
            if g_step_ % 100 == 0:
                writer.add_summary(train_summary, g_step_)
        # 验证
        sess.run(val_iterator.initializer)
        acc_sum = .0
        for j in range(10):
            acc_ = sess.run(accuracy, feed_dict={handle: val_handle,  keep_prob: 0.5, train_mode: False})
            acc_sum += acc_
        print("@@@  Validation--accuracy:{}".format(acc_sum / 10))
        log.write("@@@  Validation--accuracy:{}".format(acc_sum / 10))

        saver.save(sess, '{}/model_epoch_{}.ckpt'.format(model_path, n+1))