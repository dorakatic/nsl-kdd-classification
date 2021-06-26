from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import keras
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from sklearn.datasets import make_classification
from termcolor import colored
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import accuracy_score, classification_report, \
    plot_confusion_matrix
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
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

all_data = pd.concat([train_data, test_data])


all_data.shape
all_data.head(5)
# print(train_data.head(24))
test_data.nunique()

print(set(list(all_data['att_label'])))


train_data.describe()
print('Label distribution Training set:')
print(train_data['att_label'].value_counts())
print()
print('Label distribution Test set:')
print(test_data['att_label'].value_counts())

# normal=1, attack=0
train_attack = train_data.att_label.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_data.att_label.map(lambda a: 0 if a == 'normal' else 1)

#data_with_attack = df.join(is_attack, rsuffix='_flag')
train_data['att_flag'] = train_attack
test_data['att_flag'] = test_attack

# textual attack flags
train_att_text_label = train_data.att_label.map(
    lambda a: 'normal' if a == 'normal' else 'attack')
test_att_text_label = test_data.att_label.map(
    lambda a: 'normal' if a == 'normal' else 'attack')

train_data['text_label'] = train_att_text_label
test_data['text_label'] = test_att_text_label


# attack categories - attack classifications
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
               'processtable', 'smurf', 'teardrop', 'udpstorm']


probe_attacks = ['ipsweep', 'mscan', 'nmap',
                 'portsweep', 'saint', 'satan']


u2r_attacks = ['buffer_overflow', 'loadmodule',
               'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel']


r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf',
               'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm']


# we will use these for plotting below
attack_labels = ['normal', 'dos', 'probe', 'u2r', 'r2l']
# helper function to pass to data frame mapping


def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 'dos'
    elif attack in probe_attacks:
        # probe_attacks map to 2
        attack_type = 'probe'
    elif attack in u2r_attacks:
        # u2r attacks map to 3
        attack_type = 'u2r'
    elif attack in r2l_attacks:
        # r2l attacks map to 4
        attack_type = 'r2l'
    else:
        # normal maps to 0
        attack_type = 'normal'

    return attack_type

############################################################################################


def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks map to 2
        attack_type = 2
    elif attack in u2r_attacks:
        # u2r attacks map to 3
        attack_type = 3
    elif attack in r2l_attacks:
        # r2l attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0

    return attack_type


# map the data and join to the data set
att_cat = train_data.att_label.apply(map_attack)
train_data['att_cat'] = att_cat

test_attack_map = test_data.att_label.apply(map_attack)
test_data['att_cat'] = test_attack_map
# view the result
train_data.head()
test_data.head()

# view the result
train_data.head(24)
train_data.dtypes
train_data.info()
train_data.describe()
train_data.isnull().sum()
train_data.nunique()
train_data.shape

test_data.head(24)
test_data.dtypes
test_data.info()
test_data.describe()
test_data.isnull().sum()
test_data.nunique()


###############################################################
# DATA VISUALIZATION

sns.scatterplot(x="protocol_type", y="hot",
                hue='is_host_login', data=train_data)
plt.show()


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)
fig.suptitle(f'Attack att_label', fontsize=15)

sns.countplot(x="att_label",
              palette="OrRd_r",
              data=train_data,
              order=train_data['att_label'].value_counts().index,
              ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="att_label",
              palette="GnBu_r",
              data=test_data,
              order=test_data['att_label'].value_counts().index,
              ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()
###############################################################

# funkcija za prikaz histrograma


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

###############  histogrami integer vrijednosti   #####################


hist_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised',
             'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

plot_hist(train_data, hist_cols,
          'Distributions of Integer Features in Training Set')


hist_cols2 = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised',
              'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

plot_hist(train_data, hist_cols2,
          'Distributions of Integer Features in Testing Set')

############    histogrami float vrijednosti    ###############################

rate_cols = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
             'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(train_data, rate_cols,
          'Distributions of Rate Features in Training Set')

rate_cols2 = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(train_data, rate_cols2,
          'Distributions of Rate Features in Testing Set')

########################################################################


type_frequencies = train_data['att_flag'].value_counts()
#type_frequencies = all_data['att_flag'].value_counts()
print(type_frequencies)

normal_frequency = type_frequencies.iloc[0]
print(normal_frequency)
att_freq = type_frequencies.iloc[1]

att_label = ["Normal", "Attack"]

figure = plt.figure()
plt.pie(
    [normal_frequency, att_freq],
    explode=[0, .25],
    autopct='%1.1f%%',
    shadow=True,
)

#plt.bar(att_label, type_frequencies[1])
plt.title("Percentage for train data")
plt.show()


sns.countplot(x=type_frequencies.index, y=type_frequencies.values)
plt.title("Class balance")
plt.show()


anomalies = train_data.loc[train_data['att_flag'] == 1]
print(anomalies)

# check for null values
[col for col in train_data.columns if train_data[col].isnull().sum() > 0]

# check types
train_data.dtypes

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)
font = {"size": 10}

plt.rc('font', **font)

# histogram for attack types
grouped_att_label = train_data.groupby(
    "att_label")["att_label"].count().sort_values(ascending=False)
plt.xticks(rotation=45)


sns.barplot(x=grouped_att_label.index, y=grouped_att_label.values)
plt.title("Count of attacks and normal events")
plt.ylabel("Count")

# usporedba napada po featureima
attacks = train_data.loc[train_data['att_flag'] == 1]
print(attacks)

type_att = attacks['protocol_type'].value_counts()
print(type_att)


# histogram for protocol type
grouped_att_label_protocol = attacks.groupby("protocol_type")[
    "protocol_type"].count().sort_values(ascending=False)
plt.xticks(rotation=45)


sns.barplot(x=grouped_att_label_protocol.index,
            y=grouped_att_label_protocol.values)
plt.title("Attacks by protocol type")
plt.ylabel("Count")
plt.show()

# histogram for xx type
grouped_att_norm = train_data.groupby("text_label")[
    "text_label"].count().sort_values(ascending=False)
plt.xticks(rotation=45)
sns.barplot(x=grouped_att_norm.index, y=grouped_att_norm.values)
plt.title("att_label by protocol type")
plt.ylabel("Count")
plt.show()


# Count -->	Number of connections to the same destination host as the current connection in the past two seconds
sns.catplot(x="protocol_type", y="count", hue="text_label", data=train_data)
plt.title("Number of connections to the same \n destination host as the current \n connection in the past two seconds")
plt.show()

sns.catplot(x="duration", y="count", hue="text_label", data=train_data)
plt.title("Number of connections to the same \n destination host as the current \n connection in the past two seconds")
plt.show()

# analiza po featureu
type_att = train_data['root_shell'].value_counts()
print(type_att)


##############################################################
# correlation
corr = train_data.corr().abs()
sns.heatmap(corr)
plt.show()

columns = np.full((corr.shape[0],), True, dtype=object)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = train_data.columns[columns]
print(selected_columns)
#train_data = train_data[selected_columns]

# select least correlated
corr_matrix = train_data.corr().abs().sort_values('att_label')

leastCorrelated = corr_matrix['att_label'].nsmallest(10)
leastCorrelated = list(leastCorrelated.index)


###########################################

cor = train_data.corr().abs()
sns.heatmap(cor, xtickatt_label=True, ytickatt_label=True)
plt.show()
print(cor)

# Correlation with output variable
cor_target = abs(cor["att_flag"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
relevant_features


print(train_data[["src_bytes", "dst_bytes"]].corr())
print(train_data[["src_bytes", "land"]].corr())


#################################################################################################
#########   P   r  e  p  r  o  c  e  s  s  ######################################################
#################################################################################################

# Constant features
#[col for col in train_data.columns if train_data[col].nunique() == 1]


# removing skew from data


random_state = 42

pr_train = train_data.copy()     # copy of our train set --> preproccessed train set
pr_test = test_data.copy()       # copy of our test set --> preproccessed test set


#################################################################################################


pr_train.groupby(['num_outbound_cmds']).size()
pr_test.groupby(['num_outbound_cmds']).size()

# samo 0 vrijednost pa nam ne znaci nista --> drop

pr_train.drop('num_outbound_cmds', axis=1, inplace=True)
pr_test.drop('num_outbound_cmds', axis=1, inplace=True)


# ne trebaju nam vise TREBAJUUUU
pr_train.drop('text_label', axis=1, inplace=True)
pr_test.drop('text_label', axis=1, inplace=True)

# ne trebaju nam vise TREBAJUUUU
pr_train.drop('att_label', axis=1, inplace=True)
pr_test.drop('att_label', axis=1, inplace=True)

########################################################
pr_train.groupby(['su_attempted']).size()
pr_test.groupby(['su_attempted']).size()

# su_attempted=2 -> su_attempted=0

pr_train['su_attempted'].replace(2, 0, inplace=True)
pr_test['su_attempted'].replace(2, 0, inplace=True)
pr_train.groupby(['su_attempted']).size()
pr_test.groupby(['su_attempted']).size()

########################################################

########################################################
# normalization

norm_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root',
             'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

for col in norm_cols:
    pr_train[col] = np.log(pr_train[col]+1e-6)
    pr_test[col] = np.log(pr_test[col]+1e-6)

plot_hist(pr_train, norm_cols, 'Distributions in Processed Training Set')
plot_hist(pr_test, norm_cols, 'Distributions in Processed Testing Set')

pr_train.head()
pr_test.head(15)

########################################################
pr_train.groupby(['att_cat']).size()
pr_test.groupby(['att_cat']).size()

# skalirati kategorije napada:
# TRAIN
# 0    67352
# 1    45927
# 2    11656
# 3       43
# 4      995


# TEST
# 0    9713
# 1    7458
# 2    2421
# 3     198
# 4    2754


########################################################
# identify categorical features
print('Training set:')

for col_name in pr_train.columns:
    if pr_train[col_name].dtypes == 'object':
        unique_cat = len(pr_train[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(
            col_name=col_name, unique_cat=unique_cat))

print(pr_train['service'].value_counts().sort_values(ascending=False).head())

########################################################

# LABEL ENCODING

# categorical columns into a variable --> categorical_columns
categorical_columns = ['protocol_type', 'service', 'flag']  # , 'att_cat'

train_categorical_values = pr_train[categorical_columns]
test_categorical_values = pr_test[categorical_columns]
train_categorical_values.head()


# protocol type
unique_protocol = sorted(pr_train.protocol_type.unique())
string_pro = 'Protocol_type_'
unique_protocol2 = [string_pro + x for x in unique_protocol]
# service
unique_service = sorted(pr_train.service.unique())
string_serv = 'service_'
unique_service2 = [string_serv + x for x in unique_service]
# flag
unique_flag = sorted(pr_train.flag.unique())
string_flag = 'flag_'
unique_flag2 = [string_flag + x for x in unique_flag]

# category
unique_cat = sorted(pr_train.att_cat.unique())
string_cat = 'cat_'
unique_cat2 = [string_cat + x for x in unique_cat]


# put together
dumcols = unique_protocol2 + unique_service2 + unique_flag2  # +unique_cat2
print(dumcols)

# do same for test set
unique_service_test = sorted(pr_test.service.unique())
unique_service2_test = [string_serv + x for x in unique_service_test]

test_dumcols = unique_protocol2 + \
    unique_service2_test + unique_flag2  # +unique_cat2


train_cat_val_enc = train_categorical_values.apply(
    LabelEncoder().fit_transform)
print(train_cat_val_enc.head())
# test set
test_cat_val_enc = test_categorical_values.apply(LabelEncoder().fit_transform)
print(test_cat_val_enc.head())


########################################################
# one hot encoding

enc = OneHotEncoder()
train_cat_val_hotenc = enc.fit_transform(train_cat_val_enc)
train_cat_data = pd.DataFrame(
    train_cat_val_hotenc.toarray(), columns=dumcols)

# test set
test_cat_val_hotenc = enc.fit_transform(
    test_cat_val_enc)
test_cat_data = pd.DataFrame(
    test_cat_val_hotenc.toarray(), columns=test_dumcols)

train_cat_data.head()
test_cat_data.head()


# control
tr = (set(list(pr_train['service'])))
tst = (set(list(pr_test['service'])))
print(tr-tst)  # {'http_2784', 'urh_i', 'red_i', 'http_8001', 'aol', 'harvest'}
##

# dodajemo service u test gdje ih ima 64

train_service = pr_train['service'].tolist()
test_service = pr_test['service'].tolist()
difference = list(set(train_service) - set(test_service))
string = 'service_'
difference = [string + x for x in difference]
difference


for col in difference:
    test_cat_data[col] = 0


new_train = pr_train.join(train_cat_data)
new_train.drop('flag', axis=1, inplace=True)
new_train.drop('protocol_type', axis=1, inplace=True)
new_train.drop('service', axis=1, inplace=True)

# test data
new_test = pr_test.join(test_cat_data)
new_test.drop('flag', axis=1, inplace=True)
new_test.drop('protocol_type', axis=1, inplace=True)
new_test.drop('service', axis=1, inplace=True)

print(new_train.shape)
print(new_test.shape)

new_train.head(15)

new_test.head(15)


########################################################

# M O D E L


new_train.drop('diff_level', axis=1, inplace=True)
new_test.drop('diff_level', axis=1, inplace=True)

new_train.drop('text_label', axis=1, inplace=True)
new_test.drop('text_label', axis=1, inplace=True)

new_train.drop('att_cat', axis=1, inplace=True)
new_test.drop('att_cat', axis=1, inplace=True)


new_train.drop('att_flag', axis=1, inplace=True)
new_test.drop('att_flag', axis=1, inplace=True)

new_train.drop('att_label', axis=1, inplace=True)
new_test.drop('att_label', axis=1, inplace=True)

###############################################################
###############################################################

# build model

###############################################################
###############################################################


# logistic regression multiclass

X = new_train.drop("att_cat", axis=1)
y = new_train["att_cat"]

X_t = new_test.drop("att_cat", axis=1)
y_t = new_test["att_cat"]

print('X_train dimension= ', X.shape)
print('X_test dimension= ', X_t.shape)
print('y_train dimension= ', y.shape)
print('y_train dimension= ', y_t.shape)


lm = linear_model.LogisticRegression(
    multi_class='ovr', class_weight='balanced', solver='saga')
lm.fit(X, y)

lm.score(X_t, y_t)

print(metrics.classification_report(y_t, lm.predict(X_t)))
preds = lm.predict(X_t)

filename = 'predicted_y.txt'
np.savetxt(filename, preds, delimiter=',')

np.savetxt(filename, y_t, delimiter=',')


score = lm.score(X_t, y_t)
print('Test Accuracy Score:', score*100, '%')


###############################################################
###############################################################
###############################################################

# neural network multiclass
# multi-class classification with Keras

x = new_train.drop("att_cat", axis=1)
X = np.array(x)


y = new_train["att_cat"]

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_y.shape)

x_t = new_test.drop("att_cat", axis=1)
X_t = np.array(x_t)

y_t = new_test["att_cat"]

encoder.fit(y_t)
encoded_Y_t = encoder.transform(y_t)
dummy_y_t = np_utils.to_categorical(encoded_Y_t)

# build a model
model = Sequential()
# input shape is (features,)
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()

# hyperparametar tuning # mijenjas dense ovo di je 16 i pokusavas fit na validation setu !!!

# compile the model
model.compile(optimizer='rmsprop',
              # this is different instead of binary_crossentropy (for regular classification)
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# early stopping callback
# This callback will stop the training when there is no improvement in
# the validation loss for 10 consecutive epochs.
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=10,
                                   restore_best_weights=True)  # important - otherwise you just return the last weigths...

# now we just update our model fit call
history = model.fit(X,
                    dummy_y,
                    callbacks=[es],
                    epochs=25,  # you can set this to a big number!
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.15,
                    verbose=1)

history_dict = history.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "r" is for "solid red line"
plt.plot(epochs, acc, 'r', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()