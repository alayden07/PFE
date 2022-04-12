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


# ###extraction liste des adresses IP###
# def reader(filename):

#     with open(filename)  as f:
#         log = f.read()

#         regexp = r'\d{1,}\.\d{1,}\.\d{1,}\.\d{1,}'# r pour lire les lignes telles qu'elles sont :(tq les tabs ou les nouvelles lignes)

#         ips_list = re.findall(regexp, log)
#         return ips_list

# reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.txt')


# ###creer un fichier csv qui contient les adresses IP selon leurs frequences###
# def count(ips_list):
#     return(Counter(ips_list))


# def write_csv(counter):
#     with open('C:/Users/USER/Desktop/DS3/PFE/DataSet/output.csv','w') as csvfile:
#         writer = csv.writer(csvfile)
#         header = ['IP' , 'FREQUENCY']
#         writer.writerow(header)

#         for item in counter:
#             writer.writerow( (item, counter[item]) )

# write_csv(count(reader('C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.txt')))


def parse_str(x):
    """
    Returns the string delimited by two characters.

    Example:
        `>>> parse_str('[my string]')`
        `'my string'`
    """
    return x[1:-1]


def parse_datetime(x):
    '''
    Parses datetime with timezone formatted as:
        `[day/month/year:hour:minute:second zone]`

    Example:
        `>>> parse_datetime('13/Nov/2015:11:45:42 +0000')`
        `datetime.datetime(2015, 11, 3, 11, 45, 4, tzinfo=<UTC>)`

    Due to problems parsing the timezone (`%z`) with `datetime.strptime`, the
    timezone will be obtained using the `pytz` library.
    '''
    dt = datetime.strptime(x[1:-7], '%d/%b/%Y:%H:%M:%S')
    dt_tz = int(x[-6:-3]) * 60 + int(x[-3:-1])
    return dt.replace(tzinfo=pytz.FixedOffset(dt_tz))


data = pd.read_csv(
    'C:/Users/USER/Desktop/DS3/PFE/DataSet/txtFinal.log',
    sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
    engine='python',
    na_values='-',
    header=None,
    usecols=[0, 3, 4, 5, 6, 7, 8],
    names=['ip', 'time', 'request', 'status', 'size', 'referer', 'user_agent'],
    converters={'time': parse_datetime,
                'request': parse_str,
                'status': int,
                'size': int,
                'referer': parse_str,
                'user_agent': parse_str})


data.to_csv(r'C:/Users/USER/Desktop/DS3/PFE/DataSet/OUTPUT.csv', index = False)
