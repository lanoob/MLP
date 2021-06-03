import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    import pandas as pd
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from keras import backend as K
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dropout, Flatten, Dense
    from keras.layers import Conv1D, MaxPooling1D
    from keras import optimizers

# 3 functions for recall, precision, and f1 score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# load the 2 datasets (malicious.csv and benign.csv)
#mix them and put into the same dataframe
df = []
df2 = []
df = pd.read_csv('malicious.csv', delimiter=',', low_memory=False)
df2 = pd.read_csv('benign.csv', delimiter=',', low_memory=False)
df = df.append(df2)
df = df.sample(frac=1).reset_index(drop=True)

# split into input (X) and output (y) variables
df_array = df.values
x = df_array[:, 0:83]
y = df_array[:, 83]

#split input features and target variables into training dataset and test dataset
#test dataset = 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# one-hot-encode output labels (protocol names)
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)
dummy_y_train = np_utils.to_categorical(encoded_y_train) ##########

encoder.fit(y_test)
class_labels = encoder.classes_ 							# the name of the class labels encoded
nb_classes = len(class_labels)								# the number of different labels being trained
encoded_y_test = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded_y_test)

#build the mlp
model = Sequential()
model.add(Dense(4, input_shape=(84, 1), init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
gd_optimizer1 = optimizers.Adam(lr=0.0001)
gd_optimizer2 = optimizers.RMSprop(lr=1e-3)
gd_optimizer3 = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=gd_optimizer,
              metrics=['accuracy', f1_m, precision_m, recall_m])
# fit the model
history = model.fit(x_train, dummy_y_train, batch_size=100, epochs=30, verbose=0, validation_data=(x_test, y_test))
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
model.predict_classes()
model.evaluate()
model.predict()

print('loss: ', loss)
print('accuracy: ', accuracy)
print('f1_score: ', f1_score)
print('precision: ', precision)
print('recall: ', recall)