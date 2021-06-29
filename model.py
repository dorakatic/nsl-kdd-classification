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
from sklearn import model_selection
from sklearn import datasets
import numpy as np
import pandas as pd
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

from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import accuracy_score, classification_report, \
    plot_confusion_matrix

from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, Normalizer
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

[col for col in train_data.columns if train_data[col].isnull().sum() > 0]
train_data.nunique()

[col for col in test_data.columns if test_data[col].isnull().sum() > 0]
test_data.nunique()

#all_data = pd.concat([train_data, test_data])


# all_data.shape
# all_data.head(5)
# print(train_data.head(24))
test_data.nunique()

# print(set(list(all_data['att_label'])))

train_data.head()
train_data.describe()

# check types
train_data.dtypes

print('Label distribution Training set:')
print(train_data['att_label'].value_counts())
print()
print('Label distribution Test set:')
print(test_data['att_label'].value_counts())

train_data.info()
train_data.nunique()

test_data.info()
test_data.nunique()

########################################################
# identify categorical features
print('Training set:')

for col_name in train_data.columns:
    if train_data[col_name].dtypes == 'object':
        categ_un = len(train_data[col_name].unique())
        print("Feature '{col_name}' has {categ_un} categories".format(
            col_name=col_name, categ_un=categ_un))

########################################################

########################################################
# identify categorical features
print('Test set:')

for col_name in test_data.columns:
    if test_data[col_name].dtypes == 'object':
        categ_un = len(test_data[col_name].unique())
        print("Feature '{col_name}' has {categ_un} categories".format(
            col_name=col_name, categ_un=categ_un))

########################################################


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
###############################################################

# ovo nista
sns.scatterplot(x="protocol_type", y="hot",
                hue='is_host_login', data=train_data)
plt.show()
################


# napadi po klasama za oba seta

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


print(train_data[["att_cat", "flag"]].value_counts())

train_data[["att_cat", "flag"]].groupby(["att_cat", "flag"]).size()
test_data[["att_cat", "flag"]].groupby(["att_cat", "flag"]).size()

pd.set_option('display.max_rows', None)  # or 1000
train_data[["att_cat", "service"]].groupby(
    ["att_cat", "service"]).size().sort_values(ascending=False)
test_data[["att_cat", "service"]].groupby(
    ["att_cat", "service"]).size().sort_values(ascending=False)


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

plot_hist(test_data, hist_cols2,
          'Distributions of Integer Features in Testing Set')

############    histogrami float vrijednosti    ###############################

rate_cols = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
             'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(train_data, rate_cols,
          'Distributions of Rate Features in Training Set')

rate_cols2 = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(test_data, rate_cols2,
          'Distributions of Rate Features in Testing Set')

########################################################################

# pie za trening

type_frequencies = train_data['text_label'].value_counts()
#type_frequencies = all_data['att_flag'].value_counts()
print(type_frequencies)

normal_frequency = type_frequencies.iloc[0]
print(normal_frequency)
att_freq = type_frequencies.iloc[1]

att_label = ["Normal", "Attack"]

'#218a3f'
colors = ['#02b53e', '#e0346a']

figure = plt.figure(1)
plt.pie(
    [normal_frequency, att_freq],
    explode=[0, .10],
    autopct='%1.f%%',
    shadow=False,
    labels=att_label,
    textprops={'fontsize': 18}
)

plt.title("Train data\n")
plt.show()

#################
# pie za test

type_frequencies2 = test_data['text_label'].value_counts()


normal_frequency2 = type_frequencies2.iloc[0]

att_freq2 = type_frequencies2.iloc[1]

colors = ['#02b53e', '#e0346a']

figure = plt.figure(2)
plt.pie(
    [normal_frequency2, att_freq2],
    explode=[0, .10],
    autopct='%1.f%%',
    shadow=False,
    labels=att_label,
    textprops={'fontsize': 18}
)

plt.title("Test data\n")
plt.show()


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)
fig.suptitle(f'Histogram for protocol_type', fontsize=18)

# histogram for protocol_un type
grouped_att_norm = train_data.groupby("protocol_type")[
    "protocol_type"].count().sort_values(ascending=False)
plt.xticks(rotation=45)
sns.barplot(x=grouped_att_norm.index, y=grouped_att_norm.values, ax=ax1)

ax1.set_title('Train Set', fontsize=18)
ax1.set_xlabel('protocol_type', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)


grouped_att_norm2 = test_data.groupby("protocol_type")[
    "protocol_type"].count().sort_values(ascending=False)
plt.xticks(rotation=45)
sns.barplot(x=grouped_att_norm2.index, y=grouped_att_norm2.values, ax=ax2)

ax2.set_title('Test Set', fontsize=18)
ax2.set_xlabel('protocol_type', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)


#plt.title("Protocol type - training set")
# plt.ylabel("Count")
plt.show()


###############################################################################################
###############################################################################################
###############################################################################################


anomalies = train_data.loc[train_data['att_flag'] == 1]
print(anomalies)


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


# histogram for protocol_un type
grouped_att_label_protocol = attacks.groupby("protocol_type")[
    "protocol_type"].count().sort_values(ascending=False)
plt.xticks(rotation=45)


sns.barplot(x=grouped_att_label_protocol.index,
            y=grouped_att_label_protocol.values)
plt.title("Attacks by protocol_un type")
plt.ylabel("Count")
plt.show()


# histogram for xx type
grouped_att_norm = train_data.groupby("protocol_type")[
    "text_label"].count().sort_values(ascending=False)
plt.xticks(rotation=45)
sns.barplot(x=grouped_att_norm.index, y=grouped_att_norm.values)

plt.title("Protocol type")
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


random_state = 42

pr_train = train_data.copy()     # copy of our train set --> preproccessed train set
pr_test = test_data.copy()       # copy of our test set --> preproccessed test set


#################################################################################################


pr_train.groupby(['num_outbound_cmds']).size()
pr_test.groupby(['num_outbound_cmds']).size()

# samo 0 vrijednost pa nam ne znaci nista --> drop

pr_train.drop('num_outbound_cmds', axis=1, inplace=True)
pr_test.drop('num_outbound_cmds', axis=1, inplace=True)


# ne trebaju nam vise
pr_train.drop('text_label', axis=1, inplace=True)
pr_test.drop('text_label', axis=1, inplace=True)

# ne trebaju nam vise
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

# podaci koje je potrebno normalizirati
norm_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root',
             'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# definiranje MinMaxScalera
mms = MinMaxScaler()
pr_train[norm_cols] = pd.DataFrame(
    mms.fit_transform(pr_train[norm_cols]), columns=norm_cols
)
pr_test[norm_cols] = pd.DataFrame(
    mms.fit_transform(pr_test[norm_cols]), columns=norm_cols
)

###
standard_scaler = StandardScaler().fit(pr_train[norm_cols])

pr_train[norm_cols] = \
    standard_scaler.transform(pr_train[norm_cols])

pr_test[norm_cols] = \
    standard_scaler.transform(pr_test[norm_cols])
###
scaler = RobustScaler()
pr_train[norm_cols] = pd.DataFrame(
    scaler.fit_transform(pr_train[norm_cols]), columns=norm_cols
)
pr_test[norm_cols] = pd.DataFrame(
    scaler.fit_transform(pr_test[norm_cols]), columns=norm_cols
)
####

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
        categ_un = len(pr_train[col_name].unique())
        print("Feature '{col_name}' has {categ_un} categories".format(
            col_name=col_name, categ_un=categ_un))

print(pr_train['service'].value_counts().sort_values(ascending=False).head())


print('Test set:')

for col_name in pr_train.columns:
    if pr_test[col_name].dtypes == 'object':
        categ_un = len(pr_test[col_name].unique())
        print("Feature '{col_name}' has {categ_un} categories".format(
            col_name=col_name, categ_un=categ_un))

print(pr_test['service'].value_counts().sort_values(ascending=False).head())


########################################################

# LABEL ENCODING

# categorical columns into a variable --> categorical_columns
categorical_columns = ['protocol_type', 'service', 'flag', 'att_cat']

train_categorical_values = pr_train[categorical_columns]
test_categorical_values = pr_test[categorical_columns]
train_categorical_values.head()


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

#X_lr = new_train.drop("att_cat", axis=1)
#y_lr = new_train["att_cat"]

#X_t_lr = new_test.drop("att_cat", axis=1)
#y_t_lr = new_test["att_cat"]

encoder = LabelEncoder()
X_lr = new_train.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y_lr = new_train[['cat_dos', 'cat_normal',
                  'cat_probe', 'cat_r2l', 'cat_u2r']]

y_lr = y_lr.idxmax(1)
encoder.fit_transform(y_lr)
y_lr = encoder.transform(y_lr)


X_t_lr = new_test.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y_t_lr = new_test[['cat_dos', 'cat_normal',
                  'cat_probe', 'cat_r2l', 'cat_u2r']]

y_t_lr = y_t_lr.idxmax(1)
encoder.fit(y_t_lr)
y_t_lr = encoder.transform(y_t_lr)


print('X_train dimension= ', X_lr.shape)
print('X_test dimension= ', X_t_lr.shape)
print('y_train dimension= ', y_lr.shape)
print('y_train dimension= ', y_t_lr.shape)


lm = linear_model.LogisticRegression(
    multi_class='ovr', class_weight='balanced', solver='saga')
lm.fit(X_lr, y_lr)

lm.score(X_t_lr, y_t_lr)
print("Train score is:", lm.score(X_lr, y_lr))
print("Test score is:", lm.score(X_t_lr, y_t_lr))

print(metrics.classification_report(y_t_lr, lm.predict(X_t_lr)))
preds = lm.predict(X_t_lr)

filename = 'predicted_y.txt'
np.savetxt(filename, preds, delimiter=',')

np.savetxt(filename, y_t_lr, delimiter=',')


score = lm.score(X_t_lr, y_t_lr)
print('Test Accuracy Score:', score*100, '%')


###############################################################
###############################################################
###############################################################

# neural network multiclass
# multi-class classification with Keras

x_mclss = new_train.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)

y_mclss = new_train[['cat_dos', 'cat_normal',
                     'cat_probe', 'cat_r2l', 'cat_u2r']]

x_tmc = new_test.drop(
    ['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r'], axis=1)
y_tmc = new_test[['cat_dos', 'cat_normal', 'cat_probe', 'cat_r2l', 'cat_u2r']]


######### ###              ##################

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
###################################################


encoder.fit(y_t)
encoded_Y_t = encoder.transform(y_t)
dummy_y_t = np_utils.to_categorical(encoded_Y_t)


model = Sequential()
model.add(Dense(256, input_dim=(X.shape[1],), activation='softplus'))
# deep_model.add(Dropout(0.2))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
#deep_model.add(Dense(18, activation='softplus'))
model.add(Dense(5, activation='softmax'))


model = Sequential()
# input shape is (features,)

model.add(Dense(32, input_shape=(x_mclss.shape[1],), kernel_regularizer=l2(
    0.001), bias_regularizer=l2(0.001), activation='sigmoid'))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(16, activation='sigmoid'))
model.add(keras.layers.Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.summary()


model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999,
              amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])

# This callback will stop the training when there is no improvement in
# the validation loss for 10 consecutive epochs.
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_mclss, y_mclss, callbacks=[
                    es], epochs=15, batch_size=60, shuffle=True, validation_split=0.2, verbose=1)


# hyperparametar tuning # mijenjas dense ovo di je 16 i pokusavas fit na validation setu !!!


# evaluate the model
_, train_acc = model.evaluate(x_mclss, y_mclss, verbose=0)
_, test_acc = model.evaluate(x_tmc, y_tmc, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=10,
                                   restore_best_weights=True)  # important - otherwise you just return the last weigths...


history_dict = history.history
model.get_weights()

y_pred = model.predict(x_mclss)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# Converting one hot encoded test label to label
test = list()
for i in range(len(dummy_y_t)):
    test.append(np.argmax(dummy_y_t[i]))

a = accuracy_score(pred, test)
print('Accuracy is:', a*100)


# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


fig = plt.figure()
fig.suptitle("Adam, lr=0.0005, two hidden layers")

ax = fig.add_subplot(1, 2, 1)
ax.set_title('Cost')
ax.plot(history.history['loss'], label='Training')
ax.plot(history.history['val_loss'], label='Validation')
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.set_title('Accuracy')
ax.plot(history.history['accuracy'], label='Training')
ax.plot(history.history['val_accuracy'], label='Validation')
ax.legend()

fig.show()

###############################################################
###############################################################
###############################################################
y = new_train["att_cat"]
x = new_train.drop("att_cat", axis=1)

x_t = new_test.drop("att_cat", axis=1)
y_t = new_test["att_cat"]

seed_random = 42

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.3, random_state=seed_random)


input_size = len(x_t.columns)

deep_model = Sequential()
deep_model.add(Dense(256, input_dim=input_size, activation='softplus'))
# deep_model.add(Dropout(0.2))
deep_model.add(Dense(128, activation='relu'))
deep_model.add(Dense(64, activation='relu'))
deep_model.add(Dense(32, activation='relu'))
#deep_model.add(Dense(18, activation='softplus'))
deep_model.add(Dense(5, activation='softmax'))

deep_model.compile(loss='categorical_crossentropy',
                   optimizer=Adam(learning_rate=0.001, beta_1=0.9,
                                  beta_2=0.999, amsgrad=True),
                   metrics=['accuracy'])

y_train_econded = label_encoder.transform(y_train)
y_val_econded = label_encoder.transform(y_val)
y_test_econded = label_encoder.transform(y_t)

y_train_dummy = np_utils.to_categorical(y_train_econded)
y_val_dummy = np_utils.to_categorical(y_val_econded)
y_test_dummy = np_utils.to_categorical(y_test_econded)

deep_model.fit(x_train, y_train_dummy,
               epochs=10,
               batch_size=2500,
               validation_data=(x_val, y_val_dummy))


###################################################################
#####################################################################

df_train = new_train.copy()     # copy of our train set --> preproccessed train set
df_test = new_test.copy()

X = df_train.drop("att_cat", axis=1)
Y_train = df_train["att_cat"]

sc = MinMaxScaler()
X_train = sc.fit_transform(X)

X_t = df_test.drop("att_cat", axis=1)
Y_test = df_test["att_cat"]

sc = MinMaxScaler()
X_test = sc.fit_transform(X_t)


def create_ann():
    model = Sequential()

    # here 30 is output dimension
    model.add(Dense(64, input_dim=(
        X.shape[1]), activation='relu', kernel_initializer='random_uniform'))

    # in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(8, activation='sigmoid',
              kernel_initializer='random_uniform'))

    # 5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(5, activation='softmax'))

    # loss is categorical_crossentropy which specifies that we have multiple classes

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# Since,the dataset is very big and we cannot fit complete data at once so we use batch size.
# This divides our data into batches each of size equal to batch_size.
# Now only this number of samples will be loaded into memory and processed.
# Once we are done with one batch it is flushed from memory and the next batch will be processed.
model7 = KerasClassifier(build_fn=create_ann, epochs=10,
                         batch_size=64)  # 100 -> 10
start = time.time()
model7.fit(X_train, Y_train.values.ravel())
end = time.time()
print('Training time')
print((end-start))


start_time = time.time()
Y_test_pred7 = model7.predict(X_test)
end_time = time.time()
print("Testing time: ", end_time-start_time)

start_time = time.time()
Y_train_pred7 = model7.predict(X_train)
end_time = time.time()
accuracy_score(Y_train, Y_train_pred7)


Y_train_pred7 = model7.predict(X_train)
accuracy_score(Y_test, Y_test_pred7)

###################################################################
#####################################################################

df_train = new_train.copy()     # copy of our train set --> preproccessed train set
df_test = new_test.copy()


model5 = LogisticRegression(max_iter=1200000)
model5 = linear_model.LogisticRegression(
    multi_class='ovr', class_weight='balanced', solver='saga', penalty='l2', max_iter=1000)

X = df_train.drop("att_cat", axis=1)
Y_train = df_train["att_cat"]

sc = MinMaxScaler()
X_train = sc.fit_transform(X)

X_t = df_test.drop("att_cat", axis=1)
Y_test = df_test["att_cat"]

sc = MinMaxScaler()
X_test = sc.fit_transform(X_t)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


model5.fit(X_train, Y_train.ravel())

Y_test_pred5 = model5.predict(X_test)

print("Train score is:", model5.score(X_train, Y_train))
print("Test score is:", model5.score(X_test, Y_test))


print(metrics.classification_report(Y_test, model5.predict(X_test)))
