import pandas as pd
from pandas import Series, DataFrame
import glob
from os import path
import re
import shutil
import glob
import gzip
import os
import numpy as np
from collections import Counter
import csv
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import woothee
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, GRU
from keras.layers import Bidirectional
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

###################################extraction des fichiers compréssés##############################################


# choix dossiers des fichiers apache access logs à traiter(input)
# choix dossiers des fichiers apache access logs à traiter(input)
Access_INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/dataset1'
# INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/data_sfm/data_app1/ACCESS log'
# INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/data_sfm/data_app2/ACCESS log'
# INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/data_sfm/data_app3/ACCESS log'
# choix dossier des fichiers apache error logs à traiter(input)
Error_INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/data_sfm/data_app1/Error log'

# choix du dossier de l'output des fichiers apache access log
Access_OUTPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/OutputAccess'
# choix du dossier de l'output des fichiers apache error log
Error_OUTPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/OutputError'

# choix de l'extension des fichiers compréssés
GZIP_EXTENSION = '.gz'


def Decompression_GZ(output_directory, zipped_name):
    name_without_gzip_extension = zipped_name[:-len(GZIP_EXTENSION)]  # enlèvement de '.gz'
    return os.path.join(output_directory, name_without_gzip_extension)


for file in os.scandir(Access_INPUT_DIRECTORY):
    if not file.name.lower().endswith(GZIP_EXTENSION):
        continue

    output_path = Decompression_GZ(Access_OUTPUT_DIRECTORY, file.name)

    print('Decompressing', file.path, 'to', output_path)

    with gzip.open(file.path, 'rb') as file:
        with open(output_path, 'wb') as output_file:
            output_file.write(file.read())  # ce qui se trouve dans le compteur "file" on va l'ecrire dans outputfile

for file in os.scandir(Error_INPUT_DIRECTORY):
    if not file.name.lower().endswith(GZIP_EXTENSION):
        continue

    output_path = Decompression_GZ(Error_OUTPUT_DIRECTORY, file.name)

    print('Decompressing', file.path, 'to', output_path)

    with gzip.open(file.path, 'rb') as file:
        with open(output_path, 'wb') as output_file:
            output_file.write(file.read())
        # ###transformation en txt###
# files=glob.glob('C:/Users/USER/Desktop/DS3/PFE/DataSet/dataset3/*')

# for i in files:
#     new_name=i.split('_')[0]    #fetch the name before '_'
#     write_file=open(new_name+'.txt','a')  #open file in append mode
#     read_file=open(i)
#     lines=read_file.read()
#     write_file.write(lines)
#     write_file.close()
#     read_file.close()


############################################Fusionnement fichiers decompréssés###########################################
files2 = glob.glob('C:/Users/USER/Desktop/DS3/PFE/DataSet/OutputAccess/*')
print(files2)

list1 = [i for i in files2]

with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal_Access.log', 'w') as outfile:
    for names in list1:
        with open(names) as infile:
            outfile.write(infile.read())

        outfile.write("\n")  # txtfinal.log c'est le resultat de fusionnement

files2 = glob.glob('C:/Users/USER/Desktop/DS3/PFE/DataSet/OutputError/*')
print(files2)

list1 = [i for i in files2]

with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal_Error.log', 'w') as outfile:
    for names in list1:
        with open(names) as infile:
            outfile.write(infile.read())

        outfile.write("\n")  # txtfinal.log c'est le resultat de fusionnement


# ###extraction liste des adresses IP###
# def reader(filename):
#     with open(filename) as f:
#         log = f.read()

#         regexp = r'\d{1,}\.\d{1,}\.\d{1,}\.\d{1,}'  # r pour lire les lignes telles qu'elles sont :(tq les tabs ou les nouvelles lignes)

#         ips_list = re.findall(regexp, log)
#         return ips_list


# reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log')


# ###creer un fichier csv qui contient les adresses IP selon leurs frequences###
# def count(ips_list):
#     return (Counter(ips_list))


# def write_csv(counter):
#     with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/output1.csv', 'w') as csvfile:
#         writer = csv.writer(csvfile)
#         header = ['IP', 'FREQUENCY']
#         writer.writerow(header)

#         for item in counter:
#             writer.writerow((item, counter[item]))


# write_csv(count(reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log')))


######fonction pour enlever les accolades pour string#######
def parse_str(x):
    return x[1:-1]


######fonction pour dépouiller la date######
def parse_datetime(x):
    dt = datetime.strptime(x[1:-7],
                           '%d/%b/%Y:%H:%M:%S')  # strtime est predefinie / x[1:-7] : c à d qu'on a négligé les accolades

    dt_tz = int(x[-6:-3]) * 60 + int(x[-3:-1])
    return dt.replace(tzinfo=pytz.FixedOffset(dt_tz))  # adaptation au fuseau horaire (tz: Time Zone)


def parse_pid(x):
    return x[5:-1]


######fonction pour dépouiller la date######
def parse_datetime2(x):
    dt = datetime.strptime(x[1:-1], '%a %b %d %H:%M:%S.%f %Y')

    return dt


#### creation de la dataframe des fichiers Error log
error_data = pd.read_csv(
    'C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal_Error.LOG',
    sep=r'\s(\[[^\]]+\]) (\[[^\]]+\]) (\[[^\]]+\]) (.*)(?![^\[]*\])$',
    engine='python',
    na_values='-',
    header=None,
    usecols=[0, 2, 3, 4],
    names=['Time', 'Pid', 'IP Client', 'Message'],
    converters={'Time': parse_datetime2,
                'Pid': parse_pid,
                'IP Client': parse_str,

                }
)

error_data = error_data.sort_values(by="Time")
error_data.to_csv(r'C:/Users/USER/Desktop/DS3/PFE/DataSet/Error_OUTPUT.csv', index=False)

#### creation de la dataframe fichiers Access log
access_data = pd.read_csv(
    'C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal_Access.LOG',
    sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',  # \s : whitespace
    engine='python',
    na_values='-',  # valeurs Nan
    header=None,
    # attribuer automatiquement la première ligne de data (qui correspond aux noms de colonnes réels) à la première ligne
    usecols=[0, 3, 4, 5, 6, 7, 8],  # eliminer les 2 tirets qui se trouvent après l'@ IP .
    names=['ip', 'Time', 'request', 'status', 'size', 'referer', 'user_agent'],
    converters={'Time': parse_datetime,
                'request': parse_str,
                'status': int,
                'size': int,
                'referer': parse_str,
                'user_agent': parse_str})

#####Labelisation des données dataframe access   : 1 pour les donnees qui presentent une erreur , 0 sinn
# error_label est le nom de la nouvelle colonne des labels
access_data['error_label'] = ""
for index, row in access_data.iterrows():
    if (399 < access_data['status'][index] < 499):
        access_data['error_label'][index] = 1
    else:
        access_data['error_label'][index] = 0

# compter le nombre d'echantillons de labels 1 et de labels 0
k = DataFrame(access_data.groupby(['error_label']).size().index)
k['count'] = access_data.groupby(['error_label']).size().values
print(k)

# trier selon la date et exportation sous forme d'un fichier csv
access_data = access_data.sort_values(by="Time")
access_data.to_csv(r'C:/Users/USER/Desktop/DS3/PFE/DataSet/Access_OUTPUT.csv', index=False)

# verfication de valeurs manquantes
access_data.isnull().sum()

# user agent confirmation
userAgent = DataFrame(access_data.groupby(['user_agent']).size().index)
userAgent['count'] = access_data.groupby(['user_agent']).size().values
userAgent

# status confirmation
d = DataFrame(access_data.groupby(['status']).size().index)
d['count'] = access_data.groupby(['status']).size().values
d


# regroupe les éléments qui forment une minorité (1 % ou moins) en "autres".
def replace_df_minors_with_others(df_intial,
                                  column_name):  # df_initial : dataframe avant modification , column_name: colonne concernée par la visualization
    elm_num = 1
    for index, row in df_intial.sort_values([column_name], ascending=False).iterrows():
        if (row[column_name] / df_intial[
            column_name].sum()) > 0.01:  # pourcentage d'une certaine valeur par rapport à la somme de touts les valeurs de cette colonne
            elm_num = elm_num + 1
    # df_after est la nouvelle dataframe sans les petites valeurs
    df_after = df_intial.sort_values([column_name], ascending=False).nlargest(elm_num,
                                                                              columns=column_name)  # nlargest renvoie les n premières lignes triées par colonnes dans l'ordre décroissant.
    # regroupement des petites valeurs dans "others"
    df_after.loc[len(df_intial)] = ['others', df_intial.drop(df_after.index)[
        column_name].sum()]  # .loc accéde à un groupe de lignes et de colonnes par libellé(s) ou tableau booléen.
    return df_after


# regroupe les éléments qui forment une minorité (1 % ou moins) en "autres". ( pour les dictionnaires )
def replace_dict_minors_with_others(dict_initial):
    dict_after = {}
    others = 0
    total = sum(dict_initial.values())
    for key in dict_initial.keys():
        if (dict_initial.get(key) / total) > 0.01:
            dict_after[key] = dict_initial.get(key)
        else:
            others = others + dict_initial.get(key)
    dict_after = {k: v for k, v in sorted(dict_after.items(), reverse=True, key=lambda item: item[1])}
    dict_after['others'] = others
    return dict_after


# analyser user agent avec woothee et suppression de la partie facultative

ua_counter = {}  # ua : variable userAgent
os_counter = {}  #

for index, row in userAgent.sort_values(['count'],
                                        ascending=False).iterrows():  # for index, row pour parcourir un dataframe.
    ua = woothee.parse(row['user_agent'])  # woothee est un analyseur(parser en anglais) de chaînes d'agent utilisateur.
    uaKey = ua.get('name') + ' (' + ua.get(
        'version') + ')'  # uaKey est une valeur contenant le nom et la version de chaque user agent

    # assurer l'ajout d'un nouvel user agent seulement une fois.
    if not uaKey in ua_counter:
        ua_counter[uaKey] = 0
    ua_counter[uaKey] = ua_counter[uaKey] + 1
    osKey = ua.get('os') + ' (' + ua.get(
        'os_version') + ')'  # osKey est une valeur contenant le systeme d'exploitaion et la version de chaque OS

    # assurer l'ajout d'un nouvel user agent seulement une fois.
    if not osKey in os_counter:
        os_counter[osKey] = 0
    os_counter[osKey] = os_counter[osKey] + 1

# pie chart client OS
plt.figure(figsize=(40, 10))
plt.title('Client OS')
os_counter_with_others = replace_dict_minors_with_others(os_counter)
patches, texts = plt.pie(os_counter_with_others.values(), startangle=90)
plt.legend(patches, os_counter_with_others, loc="upper left")
plt.show()

# pie chart User Agent
plt.figure(figsize=(50, 10))
plt.title('User Agent')
ua_counter_with_others = replace_dict_minors_with_others(ua_counter)
patches2, texts = plt.pie(ua_counter_with_others.values(), startangle=90)
plt.legend(patches2, ua_counter_with_others, loc="upper right")
plt.show()

# #visualisation access vs time
# plt.figure(figsize = (15, 5))
# access = access_data['request']
# access.index = access_data['time']
# access = access.resample('S').count()
# access.index.name = 'Time'
# access.plot()
# plt.title('Total Access')
# plt.ylabel('Access')
# plt.show()


# visualisation du code de réponse
plt.figure(figsize=(10, 10))
plt.pie(access_data.groupby([access_data['status'] // 100]).count().Time, counterclock=False, startangle=90)

labels = ['400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410']
patches3, texts = plt.pie(labels, startangle=90)

plt.legend(patches3, labels, loc="best")
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.pie(access_data.groupby(['status']).count().Time, counterclock=False, startangle=90, radius=0.7)

centre_circle = plt.Circle((0, 0), 0.4, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Error Status Code')
plt.show()

# Visualization taille vs status

plt.figure(figsize=(15, 5))
plt.title('size vs. status')
plt.scatter(access_data['size'] / 1000, access_data['status'], marker='.')
plt.xlabel('Size(KB)')
plt.ylabel('status')
plt.grid()
plt.show()

# division de la dataframe label_access :etiquettes/ data_acess:donnees sans etiquettes
label_access = access_data['error_label']
data_access_without_labels = access_data['request'].astype(str)

# data_access_without_labels['request'] = data_access_without_labels['request']

max_words = 1000
max_len = 150
# toknizer:tokenization basically refers to splitting up a larger body of text into smaller lines, words or even creating words for a non-English language
tok = Tokenizer(num_words=max_words)  # text to numeric
tok.fit_on_texts(data_access_without_labels)
# affectation DES SCORES aux mots
sequences = (tok.texts_to_sequences(data_access_without_labels))  # astype(str) car float has no attribute'lower'

# mettre les donnees en mm longuer de vecteur
# padding pour le remplissage des cases manquantes avec des 0
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len,
                                          padding='post')  # post c a d remplir avec des 0 A LA FIN ET NON PAS AU DEBUT

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sequences_matrix, label_access, test_size=0.33, random_state=0)

model = Sequential()
model.add(Embedding(1000, 150, input_length=max_len))  # cree hiddenlayer
model.add(GRU(150, dropout=0.2))  # 150 :max_len
# dropout : valeur entre 0 et 1 (marge d'erreur)

# model.add(Bidirectional(GRU(150, dropout=0.5))) # meilleure accuracy avec blstm
model.add(Dense(150, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # khater binary (tetkteb sur un seul bit) output layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.int32)
    return arg


x_train = my_func(x_train)
y_train = my_func(y_train)
x_test = my_func(x_test)
y_test = my_func(y_test)
history = model.fit(x_train, y_train, batch_size=128, epochs=10,
                    validation_split=0.4)  # VALIDATION split 0.2 yaani 80% lel train w 20 lel test
accr = model.evaluate(x_test, y_test)

print('test set \n Loss:{:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))




