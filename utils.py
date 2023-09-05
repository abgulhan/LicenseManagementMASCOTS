import pandas as pd
import datetime

def parse_date(text):
    separators = ['/', '\\', '-', '.', '-']
    sep = '/'
    for separator in separators:
        if separator in text:
            sep = separator
            break
    text = ' '.join(text.split(sep))
    for format in ('%Y %m %d', '%Y %m', '%Y'):
        try:
            return datetime.datetime.strptime(text, format)
        except ValueError:
            pass
    raise ValueError('no valid date format found')

'''
Crops data betwen two dates

column_name:  Which column values to use. Leave blank to use index
'''
def crop_data(df, start_date_str, end_date_str, column_name:str=None ):
    start=None
    end=None
    if start_date_str != None:
        start = parse_date(start_date_str)
    if end_date_str != None:
        end = parse_date(end_date_str)
        
    if column_name != None:
        if start == None:
            start = df[column_name].min()
        if end == None:
            end = df[column_name].max()
        return df[df[column_name].between(start, end)]
    else:
        return df.loc[start:end]
    
'''
Reads output from lmstat command to get max licenses.
'''
def parse_max_licenses(fname, license_type = 'comsol', multiple_license_versions_behavior = 'add'):
    if license_type.lower() == 'comsol' or license_type.lower() == 'matlab':
        with open(fname, 'r') as f:
            license_max = {}
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            for l in f:
                row = [i.strip() for i in l.split(' ') if i!='']
                print(row)
                f.readline()

                name = row[0]
                version = float(row[1])
                max_amount = int(row[2])
                if max_amount == 0:
                    max_amount = 9999
                if license_max.get(name) == None:
                    license_max[name] = max_amount
                else:
                    if multiple_license_versions_behavior == 'add':
                        license_max[name] += max_amount
                    if multiple_license_versions_behavior == 'max':
                        license_max[name] = max(max_amount, license_max[name])
                    if multiple_license_versions_behavior == 'min':
                        license_max[name] = min(max_amount, license_max[name])
        return license_max
    elif license_type.lower() == 'ansys':
        license_max = {}
        with open(fname, 'r') as f:
            data = f.readlines()
        for l in data:
            if 'Users' not in l: 
                continue
            
            s = l.split(':')
            name = s[0].split(' ')[2]
            if 'Total' in l:
                max_amount = int(s[1].strip().split(' ')[2])
            else:
                max_amount = int(s[2].strip().split(' ')[0])
            
            if license_max.get(name) == None:
                license_max[name] = max_amount
            else:
                if multiple_license_versions_behavior == 'add':
                    license_max[name] += max_amount
                if multiple_license_versions_behavior == 'max':
                    license_max[name] = max(max_amount, license_max[name])
                if multiple_license_versions_behavior == 'min':
                    license_max[name] = min(max_amount, license_max[name])
        return license_max

            