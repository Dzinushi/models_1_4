import tensorflow as tf
from datasets import flowers
from tensorflow.contrib import slim
from preprocessing import inception_preprocessing
from autoencoders.dataset_np.db_np_flowers import DatabaseFlowers, Table

tf.app.flags.DEFINE_string('path', '',
                           'Path to database "flowers"')
tf.app.flags.DEFINE_integer('height', 28,
                            'Preprocess image height. Original height: 224')
tf.app.flags.DEFINE_integer('width', 28,
                            'Preprocess image width. Original width: 224')
tf.app.flags.DEFINE_string('save_path', '',
                           'Save sqlite database by input path')
FLAGS = tf.app.flags.FLAGS


def load_batch(dataset, batch_size, height=224, width=224, is_training=True):
    ######################
    # Set provider train #
    ######################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(image, height, width, is_training=is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2)
    return batch_queue


def main(_):
    flowers_train = flowers.get_split(split_name='train', dataset_dir=FLAGS.path)
    flowers_validate = flowers.get_split(split_name='validation', dataset_dir=FLAGS.path)

    batch_queue_train = load_batch(flowers_train, 1, height=FLAGS.height, width=FLAGS.width, is_training=True)
    batch_queue_validate = load_batch(flowers_validate, 1, height=FLAGS.height, width=FLAGS.width, is_training=False)

    images_train, labels_train = batch_queue_train.dequeue()
    images_validate, labels_validate = batch_queue_validate.dequeue()

    max_step_train = flowers.SPLITS_TO_SIZES['train']
    max_step_validate = flowers.SPLITS_TO_SIZES['validation']

    db = DatabaseFlowers(FLAGS.save_path)
    db.create_table()

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:

        # Insert images
        progbar = tf.keras.utils.Progbar(target=max_step_train)
        for step in range(max_step_train):
            image_np, labels_np = sess.run([images_train, labels_train])
            image_np += 1
            image_np /= 2
            db.insert(Table.flowers_train, id=step+1, image_np=image_np, label_np=labels_np)
            progbar.update(step+1)
        db.commit()

        progbar = tf.keras.utils.Progbar(target=max_step_validate)
        for step in range(max_step_validate):
            image_np, labels_np = sess.run([images_validate, labels_validate])
            image_np += 1
            image_np /= 2
            db.insert(Table.flowers_validate, id=step+1, image_np=image_np, label_np=labels_np)
            progbar.update(step+1)
        db.commit()
    db.close()


if __name__ == '__main__':
    tf.app.run()
