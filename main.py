import pandas as pd

from DataPreparation import verify_dataset, delete_irrelevant_feature, str_2_datetime
from tasks import volume_vendas, tempo_medio_entrega


def main():
	df = pd.read_excel('./input/dataset_desafio_datascience.xlsx')
	verify_dataset(df)
	df = delete_irrelevant_feature(df, 'Postal Code')
	df = delete_irrelevant_feature(df, 'Customer Name')
	df = delete_irrelevant_feature(df, 'Product Name')
	df = str_2_datetime(df, 'Order Date')
	df = str_2_datetime(df, 'Ship Date')
	tempo_medio_entrega(df)
	volume_vendas(df)


if __name__ == '__main__':
	main()
