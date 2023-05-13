import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import chain
import csv
import sys
import os
import random 
from sklearn.utils import shuffle 
from tensorflow.keras import Model, layers
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.optimizers import RMSprop 
from keras import losses 
from keras import metrics
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import BatchNormalization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from keras import backend as K
from sklearn.metrics import roc_auc_score,roc_curve,RocCurveDisplay,auc
from keras.metrics import AUC
from keras.callbacks import Callback
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
import itertools


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def plot_confusion_matrix(y_test_list, predicted_labels_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()

def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix
      
def summarize_diagnostics_acc(histories):
 ax = plt.subplot()
 plt.title('Classification Accuracy')
 ax.set_xlabel('Epoch')
 ax.set_ylabel('Accuracy')
 for i in range(len(histories)):
  x = i+1
  ax.plot(histories[i].history['accuracy'], label='$Train accuracy fold = %x$' % x)
  ax.plot(histories[i].history['val_accuracy'], label='$Validation accuracy fold = %x$' % x)
  ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.10),
          ncol=5, fancybox=True, shadow=True)
 plt.show()

def summarize_diagnostics_loss(histories):
 ax = plt.subplot(111)
 plt.title('Classification Loss')
 ax.set_xlabel('Epoch')
 ax.set_ylabel('Loss')
 for i in range(len(histories)):
    x = i+1
    ax.plot(histories[i].history['loss'], label='$Train loss fold = %x$' % x)
    ax.plot(histories[i].history['val_loss'], label='$Validation loss fold = %x$' % x)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.10),
          ncol=5, fancybox=True, shadow=True)
 plt.show()

#optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.0002)

train = pd.read_csv('/dataset/m3_J_DT_GM_GE_DM_MFPC.csv')
train_fea = train.to_numpy()
train = []

x_train = train_fea[:,0:1612]

y_train = train_fea[:,1612]

# Model configuration
#define the class name of targets 
actual_target = []
predicted_target = []

#number of folds
num_folds = 10

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
histories = []
cm_holder = []
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
# Define the metrics variables to mean
tp_per_fold = []
fp_per_fold = []
tn_per_fold = []
fn_per_fold = []
precision_per_fold = []
recall_per_fold = []
auc_per_fold = []
f1_m_per_fold = []
prc_per_fold = []


kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(x_train, y_train):
  model = Sequential()

  model.add(Flatten(input_shape=(1612, )))
  model.add(BatchNormalization())
  model.add(layers.Dense(512,activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  model.add(layers.Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  model.add(layers.Dense(32,activation='relu' ))
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  model.add(Dense(1,activation='sigmoid'))

  # Compile the model
  model.compile(loss = 'binary_crossentropy',optimizer=optimizer,metrics=[METRICS,f1_m])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  history = model.fit(x_train[train], y_train[train],  validation_split=0.1, batch_size=128, epochs=300, verbose=1)
  
  # Generate generalization metrics
  print('******test*****')
  scores = model.evaluate(x_train[test], y_train[test], verbose=1)

  # Generate prediction for the confusion matrix
  y_pred = (model.predict(x_train[test]).ravel()>0.5)+0

  #add the actual and predicted targets into the array for plot the confusion matrix
  predicted_target = np.append(predicted_target, y_pred)
  actual_target = np.append(actual_target, y_train[test])

  #Generate prediction for the roc curve
  lr_probs = model.predict(x_train[test]).ravel()
  roc_auc = roc_auc_score(y_train[test], lr_probs)
  fpr, tpr, _ = roc_curve(y_train[test], lr_probs)
  tprs.append(interp(mean_fpr, fpr, tpr))
  aucs.append(roc_auc)
  plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (fold_no, roc_auc))


  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[5]} of {scores[5]*100}%')
  acc_per_fold.append(scores[5] * 100)
  loss_per_fold.append(scores[0])

  tp_per_fold.append(scores[1])
  fp_per_fold.append(scores[2])
  tn_per_fold.append(scores[3])
  fn_per_fold.append(scores[4])
  precision_per_fold.append(scores[6])
  recall_per_fold.append(scores[7])
  auc_per_fold.append(scores[8])
  f1_m_per_fold.append(scores[10])
  prc_per_fold.append(scores[9])

  histories.append(history)

  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> TP: {np.mean(tp_per_fold)}')
print(f'> FP: {np.mean(fp_per_fold)}')
print(f'> TN: {np.mean(tn_per_fold)}')
print(f'> FN: {np.mean(fn_per_fold)}')
print(f'> Precision: {np.mean(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)}')
print(f'> AUC: {np.mean(auc_per_fold)}')
print(f'> F1: {np.mean(f1_m_per_fold)}')
print(f'> PRC: {np.mean(prc_per_fold)}')
print('------------------------------------------------------------------------')
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.10),
          ncol=5, fancybox=True, shadow=True)
# plt.legend(loc="lower right",)
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

plot_confusion_matrix(actual_target, predicted_target)
summarize_diagnostics_acc(histories)
summarize_diagnostics_loss(histories)
