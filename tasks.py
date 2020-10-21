import pandas as pd
from datetime import datetime, date, time
import numpy as np

from DataPreparation import delete_irrelevant_feature

'''['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'customer ID',
       'Segment', 'City', 'State', 'Country', 'Market', 'Region', 'Product ID',
       'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Profit',
       'Shipping Cost', 'Order Priority']'''

def volume_vendas(df):
	dfr = pd.DataFrame(columns=['Segment','Total_Order'])
	ind = 0
	df_temp = df.groupby(by=['Segment'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group,'index')
		except:
			print("index col not found")
		
		
		segment = group.loc[0, 'Segment']
		dfr.at[ind, 'Segment'] = segment
		dfr.at[ind, 'Total_Order'] = group['Quantity'].sum()
		ind = ind + 1
		
	dfr.to_csv('./output/segmento_volume.csv',index=False)
	
def tempo_medio_entrega(df):
	dfr = pd.DataFrame(columns=['Country'])
	ind = 0
	df_temp = df.groupby(by=['Country'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group, 'index')
		except:
			print("index col not found")
	
		l = group.shape[0]
		country = group.loc[0, 'Country']
		dfr.at[ind, 'Country'] = country
		dfr.at[ind, 'Market'] = group.loc[0, 'Market']
		dfr.at[ind, 'Region'] = group.loc[0, 'Region']
		dfr.at[ind, 'Category'] = group.loc[0, 'Category']
		dfr.at[ind, 'Sub-Category'] = group.loc[0, 'Sub-Category']
	
		for i in np.arange(0,l):
			group.loc[i,'Days'] = (group.loc[i,'Ship Date'] - group.loc[i,'Order Date']).days
		
		dfr.at[ind, 'Median_Delivery_days'] = group['Days'].median()
		dfr.at[ind, 'Average_Delivery_days'] = group['Days'].mean()
		dfr.at[ind, 'STD'] = group['Days'].std()
		dfr.at[ind, 'VARIANCE'] = group['Days'].var()
		ind = ind + 1
	
	dfr.to_csv('./output/tempo_medio_entrega.csv', index=False)