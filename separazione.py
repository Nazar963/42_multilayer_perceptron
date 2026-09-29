import csv

import pandas as pd


def main():
	with open("data.csv", "r", encoding="utf-8") as file:
		sample = file.read(4096)
	head = csv.Sniffer().has_header(sample)

	data = pd.read_csv("data.csv", header=0 if head else None)

	data = data.drop(columns=data.columns[0])

	data = data.sample(frac=1, random_state=51).reset_index(drop=True)

	indice = int(len(data) * 0.8)
	data_train = data.iloc[:indice]
	data_validate = data.iloc[indice:]

	data_train_stand = data_train.copy()
	data_validate_stand = data_validate.copy()

	for column in data_train.select_dtypes(include="number").columns:
		mean = data_train[column].mean()
		std = data_train[column].std()
		data_train_stand[column] = (data_train[column] - mean) / std
		data_validate_stand[column] = (data_validate[column] - mean) / std

	data_train_stand.to_csv("data_train_stand.csv", index=False)
	data_validate_stand.to_csv("data_validate_stand.csv", index=False)

if __name__ == "__main__":
	main()
