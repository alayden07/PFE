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


###extraction des fichiers GZ###

INPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/dataset1'
OUTPUT_DIRECTORY = 'C:/Users/USER/Desktop/DS3/PFE/DataSet/dataset3'
GZIP_EXTENSION = '.gz'


def Decompression_GZ(output_directory, zipped_name):
    name_without_gzip_extension = zipped_name[:-len(GZIP_EXTENSION)]
    return os.path.join(output_directory, name_without_gzip_extension)


for file in os.scandir(INPUT_DIRECTORY):
    if not file.name.lower().endswith(GZIP_EXTENSION):
        continue

    output_path = Decompression_GZ(OUTPUT_DIRECTORY, file.name)

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


###Fusionnement fichiers decompréssés###
files2 = glob.glob('C:/Users/USER/Desktop/DS3/PFE/DataSet/dataset3/*')
print(files2)

list1 = [i for i in files2]  # Creating a list of filenames

with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log', 'w') as outfile:
    for names in list1:
        with open(names) as infile:
            outfile.write(infile.read())

        outfile.write("\n")


###extraction liste des adresses IP###
def reader(filename):
    with open(filename) as f:
        log = f.read()

        regexp = r'\d{1,}\.\d{1,}\.\d{1,}\.\d{1,}'  # r pour lire les lignes telles qu'elles sont :(tq les tabs ou les nouvelles lignes)

        ips_list = re.findall(regexp, log)
        return ips_list


reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log')


###creer un fichier csv qui contient les adresses IP selon leurs frequences###
def count(ips_list):
    return (Counter(ips_list))


def write_csv(counter):
    with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/output1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ['IP', 'FREQUENCY']
        writer.writerow(header)

        for item in counter:
            writer.writerow((item, counter[item]))


write_csv(count(reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log')))


def parse_str(x):
    """
    enlever les accollades
    """
    return x[1:-1]


def parse_datetime(x):

    dt = datetime.strptime(x[1:-7], '%d/%b/%Y:%H:%M:%S')

    dt_tz = int(x[-6:-3]) * 60 + int(x[-3:-1])
    return dt.replace(tzinfo=pytz.FixedOffset(dt_tz))


data = pd.read_csv(
    'C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.LOG',
    sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
    engine='python',
    na_values='-',  # Nan
    header=None,  # automatically assign the first row of df (which is the actual column names) to the first row
    usecols=[0, 3, 4, 5, 6, 7, 8],  # eliminer les 2 tirets qui se trouvent après l'@ IP .
    names=['ip', 'time', 'request', 'status', 'size', 'referer', 'user_agent'],
    converters={'time': parse_datetime,
                'request': parse_str,
                'status': int,
                'size': int,
                'referer': parse_str,
                'user_agent': parse_str})

data = data.sort_values(by="time")

data.to_csv(r'C:/Users/USER/Desktop/DS3/PFE/DataSet/OUTPUT.csv', index=False)

# checking missing values
data.isnull().sum()



# user agent confirmation
userAgent = DataFrame(data.groupby(['user_agent']).size().index)
userAgent['count'] = data.groupby(['user_agent']).size().values
userAgent



#regroupe un petit nombre d'éléments (1 % ou moins) en "autres".
def replace_df_minors_with_others(df_intial, column_name):
    elm_num = 1
    for index, row in df_intial.sort_values([column_name], ascending=False).iterrows():
        if (row[column_name] / df_intial[column_name].sum()) > 0.01:
            elm_num = elm_num + 1

    df_after = df_intial.sort_values([column_name], ascending=False).nlargest(elm_num, columns=column_name)#nlargest renvoie les n premières lignes triées par colonnes dans l'ordre décroissant.
    df_after.loc[len(df_intial)] = ['others', df_intial.drop(df_after.index)[column_name].sum()]#.loc accéde à un groupe de lignes et de colonnes par libellé(s) ou tableau booléen.
    return df_after



#For dictionaries
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

#analyser user agent avec woothee et suppression de la partie facultative

ua_counter = {}
os_counter = {}

for index, row in userAgent.sort_values(['count'], ascending=False).iterrows(): #for index, row pour parcourir un dataframe.
    ua = woothee.parse(row['user_agent'])    #woothee est un analyseur de chaînes d'agent utilisateur.
    uaKey = ua.get('name') + ' (' + ua.get('version') + ')'
    if not uaKey in ua_counter:
        ua_counter[uaKey] = 0
    ua_counter[uaKey] = ua_counter[uaKey] + 1
    osKey = ua.get('os') + ' (' + ua.get('os_version') + ')'
    if not osKey in os_counter:
        os_counter[osKey] = 0
    os_counter[osKey] = os_counter[osKey] + 1





plt.figure(figsize = (15, 10))
plt.subplot(1,2,1)
plt.title('Client OS')
os_counter_with_others = replace_dict_minors_with_others(os_counter)
plt.pie(os_counter_with_others.values(), labels = os_counter_with_others.keys(), autopct = '%1.1f%%', shadow = True, startangle = 90)

plt.subplot(1,2,2)
plt.title('User Agent')
ua_counter_with_others = replace_dict_minors_with_others(ua_counter)
plt.pie(ua_counter_with_others.values(), labels = ua_counter_with_others.keys(), autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.show()



data['status']


d = DataFrame(data.groupby(['status']).size().index)
d['count'] = data.groupby(['status']).size().values
d





#visualisation du code de réponse
plt.figure(figsize = (10, 10))
plt.pie(data.groupby([data['status'] // 100]).count().time,  counterclock=False, startangle=90)

labels = ['200', '206', '301', '302', '304', '400', '403', '404', '408', '413', '421']
plt.pie(data.groupby(['status']).count().time, labels=labels, counterclock=False, startangle=90, radius=0.7)

centre_circle = plt.Circle((0,0),0.4, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Error Status Code')
plt.show()


#Visualization taille vs status

plt.figure(figsize = (15, 5))
plt.title('size vs. status')
plt.scatter(data['size']/1000, data['status'], marker='.')
plt.xlabel('Size(KB)')
plt.ylabel('status')
plt.grid()





