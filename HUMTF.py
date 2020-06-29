#IMPORT BASIC STUFF######
import os
import sys
sys.path.append("c:\python38\lib\site-packages")
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#########################

    #IMPORTANT VALUES#

Temp_Data_Path = r'C:\Users\ocean\Desktop\DHTDATA\Training_data.csv'
Test_Data_Path = r'C:\Users\ocean\Desktop\DHTDATA\Training_data.csv'

#defines spot model weights are saved
checkpoint_path = "training_1/cp.ckpt"



EPOCHS = 25
BATCH_SIZE = 5





##############################################

#DEFINE FUNCTIONS

#SETS UP OUR CSV FILE
def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=BATCH_SIZE, # Artificially small to make examples easier to show.
      #label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset

#SHOWS OUR DATA FOR DEBUGGING PURPOSES
def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

#PACKS DATA
def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

#Normalizes data
def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

#BETTER PACKING
class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

#RUN MAIN LINE OF CODE HERE #######################################

#Apparently this is helpful
np.set_printoptions(precision=3, suppress=True)


#SET UP HUMIDITY DATASET
H_LAB_COL = 'HumR'
H_SEL_COL = ['HumR','Hum1','Hum2','Hum3']
H_O_COL = ['Hum1','Hum2','Hum3']
H_DEF = [0.0,0.0,0.0, 0.0]

#GENERATE THE HUMIDITY RAW DATA
RAW_H_DATA = get_dataset(Temp_Data_Path,
                         label_name=H_LAB_COL,
                         select_columns=H_SEL_COL,
                         column_defaults = H_DEF
                        )

#GENERATE TEST DATA
RAW_H_TEST_DATA = get_dataset(Temp_Data_Path,
                         label_name=H_LAB_COL,
                         select_columns=H_SEL_COL,
                         column_defaults = H_DEF
                        )

#PACKAGE DATA FOR USE IN TRAINING
PACKED_H = RAW_H_DATA.map(PackNumericFeatures(H_O_COL))
#PACKAGE DATA FOR USE IN TESTING
PACKED_H_TEST = RAW_H_TEST_DATA.map(PackNumericFeatures(H_O_COL))

#Gather STATISTICS on the dataset

#READ DATA FROM WHOLE CSV
H_DESC = pd.read_csv(Temp_Data_Path, usecols=H_O_COL).describe()
#Calculate MEAN
H_MEAN = np.array(H_DESC.T['mean'])
#Calculate STD
H_STD = np.array(H_DESC.T['std'])

#use values to create normalization for the model
H_normalizer = functools.partial(normalize_numeric_data, mean=H_MEAN, std=H_STD)
H_numeric_column = tf.feature_column.numeric_column('numeric',
                                                    normalizer_fn=H_normalizer,
                                                    shape=[len(H_O_COL)]
                                                   )
H_numeric_columns = [H_numeric_column]



#CREATE OUR ACTUAL MODELS
H_model = tf.keras.Sequential([
  tf.keras.layers.DenseFeatures(H_numeric_columns),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1),
])

#Put in the actualy loss metrics
H_model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    metrics=['mae', 'mse'])

#create data saver
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period = EPOCHS
                                                 )


#final data preparation
H_train_data = PACKED_H.shuffle(500)
H_Test_data = PACKED_H_TEST


H_model.load_weights(checkpoint_path)


#start model training
H_model.fit(H_train_data,epochs=EPOCHS,
            callbacks=[cp_callback]
            )



#show final average error
loss, mae, mse = H_model.evaluate(PACKED_H_TEST)
print("Testing set Mean Abs Error: {:5.2f} Percentage Points".format(mae))









#Shows actaul performance data

#test_predictions = H_model.predict(PACKED_H_TEST).flatten()
#actual_values = tf.data.experimental.make_csv_dataset(Temp_Data_Path,batch_size=5,
#                                                      na_value="?",
#                                                      num_epochs=1,
#                                                      ignore_errors=True,
#                                                      select_columns=['HumR'],
#                                                      column_defaults = [0.0]
                                    #                 )
#print(test_predictions)
#print("\n NENENENENENENENENENENE \n")
#print(actual_values)
#a = plt.axes(aspect='equal')
#plt.scatter(actual_values, test_predictions)
#plt.xlabel('True Values [Deg]')
#plt.ylabel('Predictions [Deg]')
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)



#OOF
