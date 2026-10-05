import matplotlib.pyplot as plt
import pandas as pd


def main():
	data = pd.read_csv('data.csv')

	print(data['diagnosis'].value_counts())
	print(data.head())
	print(data.shape)

	diagnosis_counts = data[1].value_counts()
	plt.bar(diagnosis_counts.index, diagnosis_counts.values)
	plt.xlabel('Diagnosis')
	plt.ylabel('Count')
	plt.title('Diagnosis Counts')
	plt.savefig('diagnosis_counts.png')

	# plt.hist(data[2])
	# plt.xlabel("Feature 2 value")
	# plt.ylabel("Frequency")
	# plt.title("Distribution of feature 2")
	# plt.savefig("feature_2_distribution.png")

	# benigno  = data[data[1] == 'B']
	# maligno = data[data[1] == 'M']
	# plt.hist(maligno[10], label='Maligno', alpha=0.5)
	# plt.hist(benigno[10], label='Benigno', alpha=0.5)
	# plt.xlabel("Feature 2 value")
	# plt.ylabel("Frequency")
	# plt.title("Distribution of feature 2 by diagnosis")
	# plt.legend()
	# plt.savefig("feature_2_distribution_by_diagnosis.png")

if __name__ == "__main__":
	main()
