import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def HARDataset(data_name):
    x_data = np.load(f'data/{data_name}/{data_name}_x_data.npy')
    y_data = np.load(f'data/{data_name}/{data_name}_y_data.npy')
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    
    print("x_train.shape : ", x_train.shape, "y_train.shape: ", y_train.shape)
    print("x_val.shape   : ", x_val.shape,    "y_val.shape: ", y_val.shape)
    print("x_test.shape  : ", x_test.shape,   "y_test.shape: ", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized)), self.temperature)
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
 
def model_evaluation(model, history, x_test, y_classified, y_contrastive=None):
    #print("[Val Acc Max Index] :", max(range(len(history.history['val_accuracy'])), key=lambda i: history.history['val_accuracy'][i]))
    #print("[Val Loss Min Index] :", min(range(len(history.history['val_loss'])), key=lambda i: history.history['val_loss'][i]))
    #print("Max Valid Acc : ", max(history.history['val_loss']))
    if y_contrastive is not None:
        test_results = model.evaluate([x_test], [y_classified, y_contrastive])
        y_pred = model.predict([x_test])[0]
    else:
        test_results = model.evaluate(x_test, y_classified)
        y_pred = model.predict([x_test])

    print("test loss : ", test_results[0])
    print("test acc  : ", accuracy_score(y_classified, y_pred.argmax(axis=1)))
    print("f1 score  : ", f1_score(y_classified, y_pred.argmax(axis=1), average="macro"))

    matrix = confusion_matrix(y_classified, y_pred.argmax(axis=1))
    print(matrix)