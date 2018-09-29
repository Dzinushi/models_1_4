import sqlite3
import io
import numpy as np


def array_to_text(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def text_to_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, array_to_text)
sqlite3.register_converter('array', text_to_array)


"""Creating table database "flowers_np.sqlite" with table "flowers",
 column 'image' with new type "array", that could be easy converted to numpy type """


class DatabaseNumpy:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns):
        """Create table with custom name, columns and its types
        columns     -   dictionary contained column name and its type by cheme:
                        columns[string_name_column] = string_column_type"""

        # Convert dictionary to view: 'column_name column type'
        list_col_name_value = []
        for name_column in columns:
            list_col_name_value.append('{} {}'.format(name_column, columns[name_column]))
        str_col_name_value = ', '.join(list_col_name_value)

        # Create and run command
        command = 'create table {}(id int PRIMARY KEY, {})'.format(table_name, str_col_name_value)
        self.cursor.execute(command)

    def select_all(self, table_name):
        command = 'select * from {}'.format(table_name)
        self.cursor.execute(command)
        images = self.cursor.fetchone()[0]
        return images

    def select_batch_by_index(self, table_name, column_name, id, batch_size):
        list_id = list(range(id, batch_size+id))
        str_list_id = ', '.join(str(x) for x in list_id)
        command = 'select {} from {} where id in ({})'.format(column_name, table_name, str_list_id)

        data = self.cursor.execute(command)

        # Get images from db
        list_images = []
        for row in range(batch_size):
            row = data.fetchone()
            if not row:
                break
            else:
                image_np = text_to_array(row[0])
                list_images.append(image_np)

        # Packing some images to one array
        image_shape = list_images[0].shape
        images = np.zeros(shape=(batch_size, image_shape[1], image_shape[2], image_shape[3]))

        for i in range(batch_size):
            images[i, :, :, :] = list_images[i]

        return images

    def insert(self, table, id, column_name, image_np):
        command = 'insert into {} (id, {}) values (?, ?)'.format(table, column_name)
        self.cursor.execute(command, (id, image_np, ))
        return self.cursor.lastrowid

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
