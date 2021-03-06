from autoencoders.dataset_np.db_np_flowers import DatabaseFlowers, Table
from time import time

db = DatabaseFlowers('/media/w_programs/Development/Python/tf_autoencoders/datasets/flowers_np_28_28_norm.sqlite')

start = time()
data = db.select_batch_by_index(Table.flowers_train, index=1, batch_size=2)
end = time() - start

db.close()

print(end)
