import subprocess

import matplotlib.pyplot as plt
import pandas as pd


def main():
	Parametri_a = "0.01\n1000\n24,24\n"
	Parametri_b = "0.1\n1000\n24,24\n"

	try:
		subprocess.run(["python", "addestramento.py"], input=Parametri_a, text=True, check=True)
		data = pd.read_csv("bonus_history_metrics.csv")
		LS_train_a = data["LS_train"].tolist()
		LS_validate_a = data["LS_validate"].tolist()
		AS_train_a = data["AS_train"].tolist()
		AS_validate_a = data["AS_validate"].tolist()

		subprocess.run(["python", "addestramento.py"], input=Parametri_b, text=True, check=True)
		data = pd.read_csv("bonus_history_metrics.csv")
		LS_train_b = data["LS_train"].tolist()
		LS_validate_b = data["LS_validate"].tolist()
		AS_train_b = data["AS_train"].tolist()
		AS_validate_b = data["AS_validate"].tolist()
	except subprocess.CalledProcessError as e:
		print(f"Errore durante l'esecuzione di addestramento.py: {e}")
		return

	plt.plot(LS_train_a, label="Training A")
	plt.plot(LS_validate_a, label="Validation A")

	plt.plot(LS_train_b, label="Training B")
	plt.plot(LS_validate_b, label="Validation B")

	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss comparison")
	plt.legend()

	plt.savefig("multi_line_loss.png")
	plt.close()

	plt.plot(AS_train_a, label="Training A")
	plt.plot(AS_validate_a, label="Validation A")

	plt.plot(AS_train_b, label="Training B")
	plt.plot(AS_validate_b, label="Validation B")

	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy comparison")
	plt.legend()

	plt.savefig("multi_line_accuracy.png")
	plt.close()

if (__name__ == "__main__"):
	main()