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
checkpoint_path = "training_2/cp.ckpt"


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
T_LAB_COL = 'TempR'
T_SEL_COL = ['TempR','Temp1','Temp2','Temp3']
T_O_COL = ['Temp1','Temp2','Temp3']
T_DEF = [0.0,0.0,0.0, 0.0]

#GENERATE THE HUMIDITY RAW DATA
RAW_T_DATA = get_dataset(Temp_Data_Path,
                         label_name=T_LAB_COL,
                         select_columns=T_SEL_COL,
                         column_defaults = T_DEF
                        )

#GENERATE TEST DATA
RAW_T_TEST_DATA = get_dataset(Temp_Data_Path,
                         label_name=T_LAB_COL,
                         select_columns=T_SEL_COL,
                         column_defaults = T_DEF
                        )

#PACKAGE DATA FOR USE IN TRAINING
PACKED_T = RAW_T_DATA.map(PackNumericFeatures(T_O_COL))
#PACKAGE DATA FOR USE IN TESTING
PACKED_T_TEST = RAW_T_TEST_DATA.map(PackNumericFeatures(T_O_COL))

#Gather STATISTICS on the dataset

#READ DATA FROM WHOLE CSV
T_DESC = pd.read_csv(Temp_Data_Path, usecols=T_O_COL).describe()
#Calculate MEAN
T_MEAN = np.array(T_DESC.T['mean'])
#Calculate STD
T_STD = np.array(T_DESC.T['std'])

#use values to create normalization for the model
T_normalizer = functools.partial(normalize_numeric_data, mean=T_MEAN, std=T_STD)
T_numeric_column = tf.feature_column.numeric_column('numeric',
                                                    normalizer_fn=T_normalizer,
                                                    shape=[len(T_O_COL)]
                                                   )
T_numeric_columns = [T_numeric_column]



#CREATE OUR ACTUAL MODEL
T_model = tf.keras.Sequential([
  tf.keras.layers.DenseFeatures(T_numeric_columns),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1),
])

#Put in the actual loss metrics
T_model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    metrics=['mae', 'mse'])

#set up data saver
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period = EPOCHS
                                                 )



#final data preparation
T_train_data = PACKED_T.shuffle(500)
T_Test_data = PACKED_T_TEST


T_model.load_weights(checkpoint_path)

#start model training
T_model.fit(T_train_data,epochs=EPOCHS,
            callbacks=[cp_callback]
            )

#show final average error
loss, mae, mse = T_model.evaluate(PACKED_T_TEST)
print("Testing set Mean Abs Error: {:5.2f} Degrees Celcius".format(mae))









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
