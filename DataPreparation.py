import pandas as pd
from datetime import datetime


def str_2_datetime(df, field):
	order_date = df[field]
	df[field] = order_date.map(lambda x: datetime.strptime(x, '%d-%m-%Y'))
	
	return df


def verify_dataset(df):
	col = df.columns
	#print(df.shape)  # dataset shape
	for c in col:
		nan_counter = df[c].isnull().sum()
		print(c, nan_counter)  # Postal Code 41296 NaN (delete it)
	
	#print(df.dtypes)  # verify datatype for all columns


def delete_irrelevant_feature(df, field):
	'''
	Delete irrelevant features like Postal_Code (in the specific problem)
	:param df: dataframe with all data
	:param field: column to be removed
	:return: filtered dataframe
	'''
	
	df = df.drop(columns=[field])
	return df
