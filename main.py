import pandas as pd
import time
from DataPreparation import verify_dataset, delete_irrelevant_feature, str_2_datetime
from tasks import volume_vendas, tempo_medio_entrega, clientes_rentaveis_ano, distribuicao_clientes, frequencia_compra, \
	forecasting


def timer(start, end):
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))



def main():
	start = time.time()
	df = pd.read_excel('./input/dataset_desafio_datascience.xlsx')
	verify_dataset(df)
	df = delete_irrelevant_feature(df, 'Postal Code')
	#df = delete_irrelevant_feature(df, 'Customer Name')
	df = delete_irrelevant_feature(df, 'Product Name')
	df = str_2_datetime(df, 'Order Date')
	df = str_2_datetime(df, 'Ship Date')
	print(df.info())
	#tempo_medio_entrega(df)
	#volume_vendas(df)
	#clientes_rentaveis_ano(df)
	#distribuicao_clientes(df)
	#frequencia_compra(df)
	forecasting(df)
	
	end = time.time()
	print("Total Execution Time : ")
	timer(start, end)


if __name__ == '__main__':
	main()
