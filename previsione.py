import numpy as np
import pandas as pd

from addestramento import feedForward, loss


def main():
	W = []
	B = []

	try:
		data = pd.read_csv("data_validate_stand.csv")
		model = np.load("model.npz")
	except FileNotFoundError as e:
		print(e)
		return

	X = data.drop(columns=["diagnosis"]).to_numpy()
	Y = data["diagnosis"].to_numpy()
	Y = np.where(Y == 'B', 0, 1)

	topologia = model["layers"]
	n_layers = len(topologia) - 1

	for i in range(n_layers):
		w = model[f"w{i + 1}"]
		b = model[f"b{i + 1}"]
		W.append(w)
		B.append(b)
		print(W[i].shape, B[i].shape)

	print (X.shape, Y.shape)
	_, A = feedForward(X, W, B)
	# print("this is the result", A[-1])
	p = np.argmax(A[-1], axis=1)
	bce = loss(A[-1], Y)
	accuracy = np.mean(p == Y)

	print("Questa è la topologia della rete:", topologia)
	print(f"--------\nEsempi: {X.shape[0]}\nLoss: {bce:.4f}\nAccuracy: {accuracy:.4f}")

if __name__ == "__main__":
	main()