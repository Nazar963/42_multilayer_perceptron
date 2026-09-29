import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n = 0.01
epochs = 1000
hidden_layers = [24, 24]
e = 1e-15

def	ReLU(z):
	return (np.maximum(0,z))

def	ReLU_derivative(z):
	return (z > 0).astype(float)

def	softMax(z):
	z_simplified = z - np.max(z, axis=1, keepdims=True)
	z_e = np.exp(z_simplified)

	return (z_e / np.sum(z_e, axis=1, keepdims=True))

def	feedForward(X, W, B):
	Z = []
	A = []

	for i in range(len(W)):
		if i == 0:
			z = X @ W[i] + B[i]
		else:
			z = A[i-1] @ W[i] + B[i]
		a = ReLU(z) if i < len(W) - 1 else softMax(z)
		Z.append(z)
		A.append(a)

	return (Z, A)

def	loss(Y_pred, Y_train):
	Y_train_ridimensionato = np.eye(2)[Y_train]
	p = np.clip(Y_pred, e, 1 - e)
	bce = -np.mean(Y_train_ridimensionato * np.log(p) + (1 - Y_train_ridimensionato) * np.log(1 - p))

	return (bce)

def	backPropagation(X, Y, W, Z, A):
	dW = [None] * len(W)
	db = [None] * len(W)

	Y_ridimensionato = np.eye(2)[Y]
	m = X.shape[0]

	dZ = A[-1] - Y_ridimensionato

	for i in reversed(range(len(W))):

		A_previous = X if i == 0 else A[i - 1]

		dW[i] = (A_previous.T @ dZ) / m
		db[i] = np.sum(dZ, axis=0) / m

		if i > 0:
			dA_previous = dZ @ W[i].T
			dZ = dA_previous * ReLU_derivative(Z[i - 1])

	return (dW, db)

def	gradient_descent(W, B, dW, db, n):
	for i in range(len(W)):
		W[i] = W[i] - n * dW[i]
		B[i] = B[i] - n * db[i]

	return (W, B)

def	config_params():
	try:
		learning_rate = input("Enter the learning rate (default is 0.01): ")
		if learning_rate:
			global n
			n = float(learning_rate)
			if n <= 0:
				raise ValueError("Invalid input. Learning rate must be greater than 0.")

		value_epochs = input("Enter the number of epochs (default is 1000): ")
		if value_epochs:
			global epochs
			epochs = int(value_epochs)
			if epochs <= 0:
				raise ValueError("Invalid input. Number of epochs must be greater than 0.")

		value_hl = input("Enter the layers configuration (default is 24, 24): ")
		if value_hl:
			global hidden_layers
			hidden_layers = list(map(int, value_hl.split(",")))
			if (len(hidden_layers) < 2):
				print("Invalid input. There must be at least two hidden layers.")
				sys.exit(1)
			for i in range(len(hidden_layers)):
				if (hidden_layers[i] <= 0):
					print("Invalid input. The number of neurons in each layer must be greater than 0.")
					sys.exit(1)
	except ValueError as e:
		print(e)
		sys.exit(1)

def	main():
	W = []
	B = []

	LS_train = []
	LS_validate = []
	AS_train = []
	AS_validate = []

	bonus_pazienza = 10
	bonus_count = 0
	bonus_miglior_loss = float("inf")

	config_params()

	# print("Learning rate:", n)
	# print("Number of epochs:", epochs)
	# print("Hidden layers:", hidden_layers)

	try:
		data_train_stand = pd.read_csv("data_train_stand.csv")
		data_validate_stand = pd.read_csv("data_validate_stand.csv")
	except FileNotFoundError:
		print("Error: You have to execute the separazione.py")
		sys.exit(1)

	X_train = data_train_stand.drop(columns=["diagnosis"]).to_numpy()
	Y_train = data_train_stand["diagnosis"].to_numpy()
	Y_train = np.where(Y_train == 'B', 0, 1)

	X_validate = data_validate_stand.drop(columns=["diagnosis"]).to_numpy()
	Y_validate = data_validate_stand["diagnosis"].to_numpy()
	Y_validate = np.where(Y_validate == 'B', 0, 1)

	print("X_train shape:", X_train.shape)
	print("X_valid shape:", X_validate.shape)

	layers = [30] + hidden_layers + [2]
	print("Layers configuration:", layers)

	for i in range(len(layers) - 1):
		input_s = layers[i]
		output_s = layers[i + 1]
		limit = np.sqrt(6 / input_s)
		w = np.random.uniform(-limit, limit, (input_s, output_s))
		b = np.zeros(output_s)
		W.append(w)
		B.append(b)

	for epoch in range(epochs):
		Z, A = feedForward(X_train, W, B)
		_, A_validate = feedForward(X_validate, W, B)

		bce = loss(A[-1], Y_train)
		bce_validate = loss(A_validate[-1], Y_validate)

		P_train = np.argmax(A[-1], axis=1)
		P_validate = np.argmax(A_validate[-1], axis=1)

		Ac_train = np.mean(P_train == Y_train)
		Ac_validate = np.mean(P_validate == Y_validate)

		print(
			f"epoch {epoch + 1}/{epochs} "
			f"- loss: {bce:.4f} "
			f"- val_loss: {bce_validate:.4f} "
			f"- accuracy: {Ac_train:.4f} "
			f"- val_accuracy: {Ac_validate:.4f}"
		)

		AS_train.append(Ac_train)
		AS_validate.append(Ac_validate)
		LS_train.append(bce)
		LS_validate.append(bce_validate)

		if bce_validate < bonus_miglior_loss:
			bonus_miglior_loss = bce_validate
			last_best_w = [w.copy() for w in W]
			last_best_b = [b.copy() for b in B]
			bonus_count = 0
		else:
			bonus_count += 1
		if bonus_count >= bonus_pazienza:
			W = last_best_w
			B = last_best_b
			print(
				f"Early stopping at epoch {epoch + 1}"
				f" due to no improvement in validation loss for {bonus_pazienza} consecutive epochs."
			)
			break

		DW, DB = backPropagation(X_train, Y_train, W, Z, A)
		W, B = gradient_descent(W, B, DW, DB, n)

	bonus_history_metrics = pd.DataFrame({
		"AS_train": AS_train,
		"AS_validate": AS_validate,
		"LS_train": LS_train,
		"LS_validate": LS_validate
	})
	bonus_history_metrics.to_csv("bonus_history_metrics.csv", index=False)

	model_data = {}
	for i in range(len(W)):
		model_data[f"w{i + 1}"] = W[i]
		model_data[f"b{i + 1}"] = B[i]
	model_data["layers"] = layers
	print("saving './model model.npz' to disk...")
	np.savez("model.npz", **model_data)

	plt.plot(LS_train, label="Training")
	plt.plot(LS_validate, label="Validation")

	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss")
	plt.legend()

	plt.savefig("loss.png")
	plt.close()

	plt.plot(AS_train, label="Training")
	plt.plot(AS_validate, label="Validation")

	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy")
	plt.legend()

	plt.savefig("accuracy.png")
	plt.close()

if __name__ == "__main__":
	main()