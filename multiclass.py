from sklearn.metrics import f1_score
import time
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Nadam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn import linear_model
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding

from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import accuracy_score, classification_report, \

from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/AnomalyDet'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# column names for data
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "att_label", "diff_level"]


train_data = pd.read_csv(
    "C:\\Users\\Dora\\Desktop\\AnomalyDet\\train.csv", header=None, names=col_names)


test_data = pd.read_csv(
    "C:\\Users\\Dora\\Desktop\\AnomalyDet\\test.csv", header=None, names=col_names)


train_data.head()
train_data.describe()
# check types
train_data.dtypes
train_data.info()
train_data.nunique()

test_data.info()
test_data.nunique()

# attack categories - attack classifications
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
               'processtable', 'smurf', 'teardrop', 'udpstorm']


probe_attacks = ['ipsweep', 'mscan', 'nmap',
                 'portsweep', 'saint', 'satan']


u2r_attacks = ['buffer_overflow', 'loadmodule',
               'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel']


r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf',
               'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm']


# helper function formapping
def map_attack(attack):
    if attack in dos_attacks:

        attack_type = 'dos'
    elif attack in probe_attacks:

        attack_type = 'probe'
    elif attack in u2r_attacks:

        attack_type = 'u2r'
    elif attack in r2l_attacks:

        attack_type = 'r2l'
    else:

        attack_type = 'normal'

    return attack_type


# map the data and join to the data set
att_cat = train_data.att_label.apply(map_attack)
train_data['att_cat'] = att_cat

test_attack_map = test_data.att_label.apply(map_attack)
test_data['att_cat'] = test_attack_map

# view the result
train_data.head()
test_data.head()


###############################################################
# DATA VISUALIZATION

# connections by class

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)
fig.suptitle(f'Attack label', fontsize=15)

sns.countplot(x="att_cat",
              palette="OrRd_r",
              data=train_data,
              order=train_data['att_cat'].value_counts().index,
              ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('class', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="att_cat",
              palette="GnBu_r",
              data=test_data,
              order=test_data['att_cat'].value_counts().index,
              ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('class', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()

###############################################################

# protocol_type training

sns.catplot(x="att_cat", y="count", hue="protocol_type",
            data=train_data, kind="bar")
plt.legend(fontsize='x-large', title_fontsize='40')
plt.title("Protocol_type for classes")
plt.show()

# protocol_type test

sns.catplot(x="att_cat", y="count", hue="protocol_type",
            data=test_data, kind="bar")
plt.legend(fontsize='x-large', title_fontsize='40')
plt.title("Protocol_type for classes")
plt.show()

###############################################################

# flag value count by class
train_data[["att_cat", "flag"]].groupby(["att_cat", "flag"]).size()
test_data[["att_cat", "flag"]].groupby(["att_cat", "flag"]).size()

# service value count by class
pd.set_option('display.max_rows', None)
train_data[["att_cat", "service"]].groupby(
    ["att_cat", "service"]).size().sort_values(ascending=False)
test_data[["att_cat", "service"]].groupby(
    ["att_cat", "service"]).size().sort_values(ascending=False)

###############################################################

# function for histograms


def plot_hist(df, cols, title):
    grid = gridspec.GridSpec(5, 3, wspace=0.5, hspace=1)
    fig = plt.figure(figsize=(100, 25))

    for n, col in enumerate(df[cols]):
        ax = plt.subplot(grid[n])

        ax.hist(df[col], bins=20)
        #ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{col} distribution', fontsize=15)

    fig.suptitle(title, fontsize=20)
    grid.tight_layout(fig, rect=[0, 1, 1, 0.97])
    plt.show()


# integer histograms
hist_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised',
             'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

plot_hist(train_data, hist_cols,
          'Distributions of Integer Features in Training Set')


hist_cols2 = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised',
              'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

plot_hist(test_data, hist_cols2,
          'Distributions of Integer Features in Testing Set')


# float histograms
rate_cols = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
             'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(train_data, rate_cols,
          'Distributions of Rate Features in Training Set')

rate_cols2 = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(test_data, rate_cols2,
          'Distributions of Rate Features in Testing Set')


###############################################################
#  P   r  e  p  r  o  c  e  s  s


random_state = 42

pr_train = train_data.copy()     # copy of our train set --> preproccessed train set
pr_test = test_data.copy()       # copy of our test set --> preproccessed test set


pr_train.groupby(['num_outbound_cmds']).size()
pr_test.groupby(['num_outbound_cmds']).size()

# only 0 value --> drop
pr_train.drop('num_outbound_cmds', axis=1, inplace=True)
pr_test.drop('num_outbound_cmds', axis=1, inplace=True)


# no meaning for classification
pr_train.drop('att_label', axis=1, inplace=True)
pr_test.drop('att_label', axis=1, inplace=True)

pr_train.groupby(['su_attempted']).size()
pr_test.groupby(['su_attempted']).size()

# su_attempted=2 -> su_attempted=0
pr_train['su_attempted'].replace(2, 0, inplace=True)
pr_test['su_attempted'].replace(2, 0, inplace=True)
pr_train.groupby(['su_attempted']).size()
pr_test.groupby(['su_attempted']).size()


###############################################################
# normalization

# data for normalization
norm_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root',
             'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# define MinMaxScaler
mms = MinMaxScaler()
pr_train[norm_cols] = pd.DataFrame(
    mms.fit_transform(pr_train[norm_cols]), columns=norm_cols
)
pr_test[norm_cols] = pd.DataFrame(
    mms.fit_transform(pr_test[norm_cols]), columns=norm_cols
)


plot_hist(pr_train, norm_cols, 'Distributions in Processed Training Set')
plot_hist(pr_test, norm_cols, 'Distributions in Processed Testing Set')

pr_train.head()
pr_test.head(15)


###############################################################
# LABEL ENCODING

# categorical columns into a variable --> categorical_columns
categorical_columns = ['protocol_type', 'service', 'flag', 'att_cat']

train_categorical_values = pr_train[categorical_columns]
test_categorical_values = pr_test[categorical_columns]
train_categorical_values.head()

# train set
train_cat_val_enc = train_categorical_values.apply(
    LabelEncoder().fit_transform)
print(train_cat_val_enc.head())

# test set
test_cat_val_enc = test_categorical_values.apply(LabelEncoder().fit_transform)
print(test_cat_val_enc.head())


###############################################################
# one hot encoding

# protocol_un type
protocol_un = sorted(pr_train.protocol_type.unique())
string_pro = 'Protocol_type_'
protocol_str = [string_pro + x for x in protocol_un]

# service
service_un = sorted(pr_train.service.unique())
string_serv = 'service_'
service_str = [string_serv + x for x in service_un]

# flag
flag_un = sorted(pr_train.flag.unique())
string_flag = 'flag_'
flag_str = [string_flag + x for x in flag_un]

# category
categ_un = sorted(pr_train.att_cat.unique())  # pronađi sve jedinstvene klase
string_cat = 'cat_'  # dodaj prefiks "cat_"
categ_str = [string_cat + x for x in categ_un]  # npr. "cat_dos"


# put together
col_name_dummy = protocol_str + service_str + flag_str + categ_str
print(col_name_dummy)

# do same for test set
service_un_test = sorted(pr_test.service.unique())
service_str_test = [string_serv + x for x in service_un_test]

test_col_name_dummy = protocol_str + \
    service_str_test + flag_str+categ_str


enc = OneHotEncoder()
train_cat_val_hotenc = enc.fit_transform(train_cat_val_enc)
train_cat_data = pd.DataFrame(
    train_cat_val_hotenc.toarray(), columns=col_name_dummy)

# test set
test_cat_val_hotenc = enc.fit_transform(
    test_cat_val_enc)
test_cat_data = pd.DataFrame(
    test_cat_val_hotenc.toarray(), columns=test_col_name_dummy)

train_cat_data.head()
test_cat_data.head()


# control
tr = (set(list(pr_train['service'])))
tst = (set(list(pr_test['service'])))
print(tr-tst)  # {'http_2784', 'urh_i', 'red_i', 'http_8001', 'aol', 'harvest'}

# add service in test, there are 64 values
train_service = pr_train['service'].tolist()
test_service = pr_test['service'].tolist()
difference = list(set(train_service) - set(test_service))
string = 'service_'
difference = [string + x for x in difference]
difference


for col in difference:
    test_cat_data[col] = 0

# train data
new_train = pr_train.join(train_cat_data)
new_train.drop('flag', axis=1, inplace=True)
new_train.drop('protocol_type', axis=1, inplace=True)
new_train.drop('service', axis=1, inplace=True)
new_train.drop('att_cat', axis=1, inplace=True)


# test data
new_test = pr_test.join(test_cat_data)
new_test.drop('flag', axis=1, inplace=True)
new_test.drop('protocol_type', axis=1, inplace=True)
new_test.drop('service', axis=1, inplace=True)
new_test.drop('att_cat', axis=1, inplace=True)

print(new_train.shape)
print(new_test.shape)

new_train.head(15)
new_test.head(15)


###############################################################
# M O D E L


# logistic regression multiclass
X = new_train.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y = new_train[['cat_dos', 'cat_normal',
               'cat_probe', 'cat_r2l', 'cat_u2r']]

# reverse label enc for logistic regression model
encoder = LabelEncoder()
y = y.idxmax(1)
encoder.fit_transform(y)
y = encoder.transform(y)


X_t = new_test.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y_t = new_test[['cat_dos', 'cat_normal',
                'cat_probe', 'cat_r2l', 'cat_u2r']]

y_t = y_t.idxmax(1)
encoder.fit(y_t)
y_t = encoder.transform(y_t)


print('X_train dimension= ', X.shape)
print('X_test dimension= ', X_t.shape)
print('y_train dimension= ', y.shape)
print('y_train dimension= ', y_t.shape)


lm = linear_model.LogisticRegression(
    multi_class='ovr', class_weight='balanced', solver='saga', C=10)
lm.fit(X, y)

lm.score(X_t, y_t)
print("Train score is:", lm.score(X, y))
print("Test score is:", lm.score(X_t, y_t))

y_preds = lm.predict(X_t)

f1_score(y_t, y_preds, average='macro')

###############################################################
# neural network
# multi-class classification with Keras

X = new_train.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y = new_train[['cat_dos', 'cat_normal',
               'cat_probe', 'cat_r2l', 'cat_u2r']]

X_t = new_test.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)
y_t = new_test[['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r']]


model = Sequential()
# input shape is (features,)

model.add(Dense(32, input_shape=(X.shape[1],), kernel_regularizer=l2(
    0.001), bias_regularizer=l2(0.001), activation='sigmoid'))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(16, activation='sigmoid'))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.summary()


model.compile(optimizer=Adam(learning_rate=0.01, beta_1=0.99, beta_2=0.999,
              amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])


# This callback will stop the training when there is no improvement in
# the validation loss for 4 consecutive epochs.
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=3,
                                   restore_best_weights=True)


history = model.fit(X, y, callbacks=[
                    es], epochs=15, batch_size=60, shuffle=True, validation_split=0.15, verbose=1)


# evaluate the model
_, train_acc = model.evaluate(X, y, verbose=0)
_, test_acc = model.evaluate(X_t, y_t, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


history_dict = history.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
