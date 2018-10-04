from autoencoders.dataset_np.db_np import DatabaseNumpy


class Table():
        flowers_train = 'flowers_train'
        flowers_validate = 'flowers_validate'


class DatabaseFlowers(DatabaseNumpy):
    def __init__(self, dataset):
        super(DatabaseFlowers, self).__init__(dataset)
        self.height = 28
        self.width = 28
        self.num_samples = {'train': 3320, 'valid': 350}
        self.num_classes = 5

    def create_table(self):
        super(DatabaseFlowers, self).create_table(Table.flowers_train, {'image': 'text', 'label': 'text'})
        super(DatabaseFlowers, self).create_table(Table.flowers_validate, {'image': 'text', 'label': 'text'})

    def select_batch_img_by_index(self, table, index, batch_size):
        return super(DatabaseFlowers, self).select_batch_img_by_index(table, 'image', index, batch_size)

    def select_batch_by_index(self, table, index, batch_size):
        return super(DatabaseFlowers, self).select_batch_by_index(table, 'image', 'label', index, batch_size)

    def insert(self, table, id, image_np, label_np):
        if table == Table.flowers_train or table == Table.flowers_validate:
            super(DatabaseFlowers, self).insert(table=str(table), id=id, img_col='image', label_col='label', image_np=image_np, label_np=label_np)
        else:
            raise ValueError('Unknown table. Table {} not exist'. format(table))
