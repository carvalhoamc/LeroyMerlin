import pandas as pd
from datetime import datetime, date, time
import numpy as np
import matplotlib.pyplot as plt

from DataPreparation import delete_irrelevant_feature

'''['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'customer ID',
       'Segment', 'City', 'State', 'Country', 'Market', 'Region', 'Product ID',
       'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Profit',
       'Shipping Cost', 'Order Priority']'''


def volume_vendas(df):
	dfr = pd.DataFrame(columns=['Segment', 'Total_Order'])
	ind = 0
	df_temp = df.groupby(by=['Segment'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group, 'index')
		except:
			print("index col not found")
		
		segment = group.loc[0, 'Segment']
		dfr.at[ind, 'Segment'] = segment
		dfr.at[ind, 'Total_Order'] = group['Quantity'].sum()
		ind = ind + 1
	
	dfr.to_csv('./output/segmento_volume.csv', index=False)


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
		
		for i in np.arange(0, l):
			group.loc[i, 'Days'] = (group.loc[i, 'Ship Date'] - group.loc[i, 'Order Date']).days
		
		dfr.at[ind, 'Median_Delivery_days'] = group['Days'].median()
		dfr.at[ind, 'Average_Delivery_days'] = group['Days'].mean()
		dfr.at[ind, 'STD'] = group['Days'].std()
		dfr.at[ind, 'VARIANCE'] = group['Days'].var()
		ind = ind + 1
	
	dfr.to_csv('./output/tempo_medio_entrega.csv', index=False)


def clientes_rentaveis_ano(df):
	begin_year = df['Order Date'].dt.year.min()
	end_year = df['Order Date'].dt.year.max()
	years = []
	while begin_year <= end_year:
		years.append(begin_year)
		begin_year = begin_year + 1
	
	for y in years:
		dfyear = df[df['Order Date'].dt.year == y]
		dfr = pd.DataFrame(columns=['Segment'])
		ind = 0
		df_temp = dfyear.groupby(by=['Segment', 'Country', 'Category', 'Sub-Category'])
		for name, group in df_temp:
			group = group.reset_index()
			try:
				group = delete_irrelevant_feature(group, 'index')
			except:
				print("index col not found")
			
			dfr.at[ind, 'Year'] = y
			dfr.at[ind, 'Country'] = group.loc[0, 'Country']
			dfr.at[ind, 'Market'] = group.loc[0, 'Market']
			dfr.at[ind, 'Region'] = group.loc[0, 'Region']
			dfr.at[ind, 'Category'] = group.loc[0, 'Category']
			dfr.at[ind, 'Sub-Category'] = group.loc[0, 'Sub-Category']
			
			dfr.at[ind, 'Segment'] = group.loc[0, 'Segment']
			dfr.at[ind, 'Profit'] = group['Profit'].sum()
			ind = ind + 1
		
		dfr.to_csv('./output/segmento_rentavel_' + str(y) + '.csv', index=False)
		
		dfr1 = pd.DataFrame(columns=['Segment'])
		ind1 = 0
		df_temp = dfr.groupby(by=['Segment'])
		for name, group in df_temp:
			group = group.reset_index()
			try:
				group = delete_irrelevant_feature(group, 'index')
			except:
				print("index col not found")
			
			dfr1.at[ind1, 'Year'] = y
			dfr1.at[ind1, 'Segment'] = group.loc[0, 'Segment']
			dfr1.at[ind1, 'Profit'] = group['Profit'].sum()
			ind1 = ind1 + 1
		
		dfr1.to_csv('./output/segmento_rentavel_total_' + str(y) + '.csv', index=False)


def distribuicao_clientes(df):
	dfr = pd.DataFrame(columns=['Country', 'Segment', 'Clients'])
	ind = 0
	df_temp = df.groupby(by=['Country', 'Segment'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group, 'index')
		except:
			print("index col not found")
		
		dfr.at[ind, 'Country'] = group.loc[0, 'Country']
		dfr.at[ind, 'Segment'] = group.loc[0, 'Segment']
		dfr.at[ind, 'Clients'] = group['customer ID'].unique().shape[0]
		ind = ind + 1
	
	dfr.to_csv('./output/distribuicao_clientes.csv', index=False)
	df_temp = dfr.groupby(by=['Country'])
	for name, group in df_temp:
		label = list(group['Segment'])
		values = list(group['Clients'])
		pizza_graph(name, values, label)


def pizza_graph(country, values, labels):
	plt.pie(values, labels=labels, autopct='%1.1f%%')
	plt.title(country)
	plt.axis('equal')
	plt.savefig('./output/distribuicao_clientes/distribuicao_clientes_' + country + '.pdf', dpi=300,
	            bbox_inches='tight')
	plt.close()


def frequencia_compra(df):
	dfr = pd.DataFrame(columns=['customer ID'])
	ind = 0
	df_temp = df.groupby(by=['customer ID'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group, 'index')
		except:
			print("index col not found")
		
		# mesmo customer ID em varios paises
		dfr.at[ind, 'customer ID'] = group.loc[0, 'customer ID']
		dfr.at[ind, 'Frequency'] = group['Order Date'].unique().shape[0]  # data ultima compra
		dfr.at[ind, 'Profit'] = group['Profit'].sum()
		dfr.at[ind, 'Sales'] = group['Sales'].sum() #considerando que a sales = quantidade x preco unitario
		last_purchase = group['Order Date'].max()
		reference_date = datetime.strptime('31-12-2014', '%d-%m-%Y')
		recencia = (reference_date - last_purchase).days
		dfr.at[ind, 'Recencia'] = recencia
		
		if recencia == 0:
			recencia = 1
			
		dfr.at[ind,'CustomerValue'] = (dfr.at[ind, 'Frequency'] * dfr.at[ind, 'Profit'])/recencia #quanto maior o
		# tempo de ultima compra menor o valor do cliente
		ind = ind + 1
	#Profit,Sales,Recencia,CustomerValue
	dfr['Profit'] = np.round(dfr['Profit'], decimals=2)
	dfr['Sales'] = np.round(dfr['Sales'], decimals=2)
	dfr['Recencia'] = np.round(dfr['Recencia'], decimals=2)
	dfr['CustomerValue'] = np.round(dfr['CustomerValue'], decimals=2)
	
	dfr = dfr.sort_values(by=['CustomerValue'],ascending=False)
	dfr.to_csv('./output/valor_do_cliente.csv', index=False)
	df1 = dfr[dfr['Profit'] > 0]  # lucro positivo
	df2 = dfr[dfr['Profit'] <= 0]  # prejuizo
	df3 = dfr[dfr['Sales'] > 0]  # faturamento bruto
	
	x = df1['Frequency']
	y = df1['Profit']
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.scatter(x, y)
	ax.set_title('Frequency x Profit')
	ax.set_xlabel('Purchase Frequency in 4 Years')
	ax.set_ylabel('Profit by Client')
	plt.savefig('./output/frequencia_compras_lucro.pdf', dpi=300, bbox_inches='tight')
	plt.close()
	
	x = df2['Frequency']
	y = df2['Profit']
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.scatter(x, y)
	ax.set_title('Frequency x Profit')
	ax.set_xlabel('Purchase Frequency in 4 Years')
	ax.set_ylabel('Profit by Client')
	plt.savefig('./output/frequencia_compras_prejuizo.pdf', dpi=300, bbox_inches='tight')
	plt.close()
	
	x = df3['Frequency']
	y = df3['Sales']
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.scatter(x, y)
	ax.set_title('Frequency x Sales')
	ax.set_xlabel('Purchase Frequency in 4 Years')
	ax.set_ylabel('Sales by Client')
	plt.savefig('./output/frequencia_compras_faturamento_bruto.pdf', dpi=300, bbox_inches='tight')
	plt.close()
	
	
