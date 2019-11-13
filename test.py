import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

# le o CSV
CSV = 'results_csgo_201911130043.csv'
dataframe = pd.read_csv(CSV)
dataframe.head()

# separa a base para treino, teste e validação
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


# converte o dataframe gerado pelo pandas para um dataset do tensorflow
def df_to_dataset(dataframe, shuffle=True):
    dataframe = dataframe.copy()

    # tira o atributo classe
    labels = dataframe.pop('winner')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    # embaralha o dataset
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(len(dataframe))
    return ds


feature_columns = []

# colunas numericas
for header in ['kill_death_ratio', 'kill_death_ratio_rival', 'win_stats', 'win_stats_rival', 'world_rank_position', 'world_rank_position_rival']:
    feature_columns.append(feature_column.numeric_column(header))

# colunas de texto
map_hashed = feature_column.categorical_column_with_hash_bucket('map_name', hash_bucket_size=1000)
map = feature_column.indicator_column(map_hashed)
feature_columns.append(map)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
