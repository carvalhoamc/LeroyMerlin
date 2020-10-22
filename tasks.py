from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from fbprophet import Prophet
from sklearn.metrics import mean_squared_log_error
from fbprophet.plot import plot_plotly, plot_components_plotly
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
		dfr.at[ind, 'Sales'] = group['Sales'].sum()  # considerando que a sales = quantidade x preco unitario
		last_purchase = group['Order Date'].max()
		reference_date = datetime.strptime('31-12-2014', '%d-%m-%Y')
		recencia = (reference_date - last_purchase).days
		dfr.at[ind, 'Recencia'] = recencia
		
		if recencia == 0:
			recencia = 1
		
		dfr.at[ind, 'CustomerValue'] = (dfr.at[ind, 'Frequency'] * dfr.at[ind, 'Profit']) / recencia  # quanto maior o
		# tempo de ultima compra menor o valor do cliente
		ind = ind + 1
	# Profit,Sales,Recencia,CustomerValue
	dfr['Profit'] = np.round(dfr['Profit'], decimals=2)
	dfr['Sales'] = np.round(dfr['Sales'], decimals=2)
	dfr['Recencia'] = np.round(dfr['Recencia'], decimals=2)
	dfr['CustomerValue'] = np.round(dfr['CustomerValue'], decimals=2)
	
	dfr = dfr.sort_values(by=['CustomerValue'], ascending=False)
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


def smape(a, f):
	return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def mape(actual, pred):
	actual, pred = np.array(actual), np.array(pred)
	return np.mean(np.abs((actual - pred) / actual)) * 100
	
# function to build one model
def build_model(pars):
	wseas, mseas, yseas, s_prior, h_prior, c_prior = pars
	m = Prophet(growth='linear',
	            daily_seasonality=False,
	            weekly_seasonality=False,
	            yearly_seasonality=False,
	            seasonality_prior_scale=s_prior,
	            holidays_prior_scale=h_prior,
	            changepoint_prior_scale=c_prior
	            )
	
	m = m.add_seasonality(
			name='weekly',
			period=7,
			fourier_order=wseas)
	
	m = m.add_seasonality(
			name='monthly',
			period=30.5,
			fourier_order=mseas)
	
	m = m.add_seasonality(
			name='yearly',
			period=365.25,
			fourier_order=yseas)
	
	return m

def prepare_for_forecasting(df):
	dfr = pd.DataFrame(columns=['DS'])
	ind = 0
	df_temp = df.groupby(by=['Order Date'])
	for name, group in df_temp:
		group = group.reset_index()
		try:
			group = delete_irrelevant_feature(group, 'index')
		except:
			print("index col not found")
		
		dfr.at[ind, 'DS'] = group.loc[0, 'Order Date']
		consumer = group[group['Segment'] == 'Consumer']
		corporate = group[group['Segment'] == 'Corporate']
		homeoffice = group[group['Segment'] == 'Home Office']
		dfr.at[ind, 'TotalSales_Consumer'] = consumer['Sales'].sum()
		dfr.at[ind, 'TotalSales_Corporate'] = corporate['Sales'].sum()
		dfr.at[ind, 'TotalSales_HomeOffice'] = homeoffice['Sales'].sum()
		ind = ind + 1
	
	reference_date = datetime.strptime('01-12-2014', '%d-%m-%Y')
	df_teste = dfr[dfr['DS'] >= reference_date]  # teste
	df_train = dfr[dfr['DS'] < reference_date]  # treinamento
	train = pd.DataFrame(columns=['ds', 'y'])
	train['ds'] = df_train['DS'].copy()
	train['y'] = df_train['TotalSales_Consumer'].copy()
	test = pd.DataFrame(columns=['ds', 'y'])
	test['ds'] = df_teste['DS'].copy()
	test['y'] = df_teste['TotalSales_Consumer'].copy()
	
	return train,test,reference_date

def modelo_profeta(train,test,reference_date):
	
	params = [[3, 5, 10, 0.5, 0.5, 0.5],
	          [15, 25, 50, 30, 15, 27],
	          [20, 30, 60, 10, 10, 10],
	          [2, 4, 8, 50.8, 5, 7],
	          [50, 100, 200, 7, 14, 21],
	          [20, 40, 85, 5.8, 0.5, 70],
	          [30,35,20,55,15,20],
	          [2, 2, 4, 15.8, 1.8, 4.8]]
	
	best_error = 10
	best_params = ()
	best_val_forecast = 0
	
	for pars in params:
		m = build_model(pars)
		m.fit(train)
		
		future = m.make_future_dataframe(periods=31, freq='D')
		future.tail()
		forecast = m.predict(future)
		
		curerror = smape(forecast['yhat'], test['y'])
		
		if curerror < best_error:
			best_error = smape(forecast['yhat'], test['y'])
			best_params = pars
			best_val_forecast = forecast
	
	print(best_params)
	m = build_model(best_params)
	m.fit(train)
	
	future = m.make_future_dataframe(periods=31, freq='D')
	future.tail()
	forecast = m.predict(future)
	forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
	forecast.to_csv('./output/forecasting_' + 'TotalSales_Consumer' + '.csv', header=True, index=False, index_label='index')
	df_predito = forecast[['ds', 'yhat']]
	df_predito = df_predito[df_predito['ds'] >= reference_date]
	df_predito = df_predito.reset_index()
	try:
		df_predito = delete_irrelevant_feature(df_predito, 'index')
	except:
		print("index col not found")
	
	test = test.reset_index()
	try:
		test = delete_irrelevant_feature(test, 'index')
	except:
		print("index col not found")
		
	dfresultados = pd.DataFrame(columns=['Data', 'Vendas reais', 'Forecast vendas', 'MAPE', 'SMAPE', 'RMSLE'])
	l = df_predito.shape[0]
	for j in np.arange(0, l):
		dfresultados.at[j, 'Data'] = test.loc[j, 'ds']
		dfresultados.at[j, 'Vendas reais'] = test.loc[j, 'y']
		dfresultados.at[j, 'Forecast vendas'] = df_predito.loc[j, 'yhat']
	
	real = dfresultados['Vendas reais'].values
	predito = dfresultados['Forecast vendas'].values
	
	SMAPE = smape(real, predito)
	print('SMAPE: ', SMAPE)
	MAPE = mape(real, predito)
	print('MAPE: ', MAPE)
	RMSLE = mean_squared_log_error(real, predito)
	print('RMSLE: ', RMSLE)
	dfresultados['MAPE'] = MAPE
	dfresultados['SMAPE'] = SMAPE
	dfresultados['RMSLE'] = RMSLE #best is 0
	
	dfresultados.to_csv('./output/resultados_forecasting' + 'TotalSales_Consumer' + '.csv', header=True, index=False,
	                    index_label='index')
	
	
	plotly.offline.plot(
			{'data':
				 [go.Scatter(x=train['ds'], y=train['y'], name='real'),
				  go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='predito'),
				  go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none',
				             name='máximo_pred'),
				  go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none',
				             name='mínimo_pred'),
				  go.Scatter(x=forecast['ds'], y=forecast['trend'], name='tendencia')],
			 'layout': {'title': 'Forecast Sales ' + 'TotalSales_Consumer', 'font': dict(family='Comic Sans MS', size=16)}},
			auto_open=False, image='png', image_filename='TotalSales_Consumer',
			output_type='file', image_width=800, image_height=600, filename='./output/forecasting_' + 'TotalSales_Consumer' + '.html',
			validate=False
	)

def modelo_AR(train,test,reference_date):
	# AR example
	from statsmodels.tsa.ar_model import AutoReg
	data = train['y']
	# fit model
	model = AutoReg(data,lags=1)
	ar_res = model.fit()
	# make prediction
	yhat = ar_res.predict(start=reference_date, end=datetime(2014,12,31))
	print(yhat)
	