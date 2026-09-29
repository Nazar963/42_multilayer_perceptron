import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n = 0.01
epochs = 1
hidden_layers = [2, 2]
e = 1e-15

def	ReLU(z):
	return (np.maximum(0,z))

def	ReLU_derivative(z):
	return (z > 0).astype(float)

def	softMax(z):
	z_simplified = z - np.max(z, axis=1, keepdims=True)
	# print("this is z_simplified::", z_simplified)
	z_e = np.exp(z_simplified)
	# print("this is z_simplified::", z_e)
	# sys.exit(1)
	return (z_e / np.sum(z_e, axis=1, keepdims=True))

def	feedForward(X, W, B):
	Z = []
	A = []
	# print("this is X", type(X))
	# print("this is W[i]", type(W[0]))
	# print("this is B[i]", type(B))
	for i in range(len(W)):
		if i == 0:
			z = X @ W[i] + B[i]
			# print("this is X", X)
			# print("this is W[i]", W[i])
			# print("this is B[i]", B[i])
			# sys.exit(1)
		else:
			z = A[i-1] @ W[i] + B[i]
			# print("this is the z", z)
		a = ReLU(z) if i < len(W) - 1 else softMax(z)
		# print("this is a", a)

		Z.append(z)
		A.append(a)
	# sys.exit(1)
	return (Z, A)

def	loss(Y_pred, Y_train):
	# print("this is Y_train:", Y_train)
	# print("this is Y_pred:", Y_pred)
	# sys.exit(1)
	Y_train_ridimensionato = np.eye(2)[Y_train]
	p = np.clip(Y_pred, e, 1 - e)
	# print("this is p:", p)
	# sys.exit(1)

	# print("this is log examples", -np.mean(Y_train_ridimensionato * np.log(p) + (1 - Y_train_ridimensionato) * np.log(1 - p)))
	# sys.exit(1)

	bce = -np.mean(Y_train_ridimensionato * np.log(p) + (1 - Y_train_ridimensionato) * np.log(1 - p))
	# print("this is log examples", bce)
	# sys.exit(1)

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

	return dW, db

def	gradient_descent(W, B, dW, db, n):
	print("this is W[0]", W[2] - n * dW[2])
	print("this is dW[0]", n * dW[2])
	sys.exit(1)
	for i in range(len(W)):
		W[i] = W[i] - n * dW[i]
		B[i] = B[i] - n * db[i]

	return W, B

def	config_params():
	try:
		learning_rate = input("Enter the learning rate (default is 0.01): ")
		if learning_rate:
			global n
			n = float(learning_rate) if (float(learning_rate) > 0) else 0.01

		value_epochs = input("Enter the number of epochs (default is 1000): ")
		if value_epochs:
			global epochs
			epochs = int(value_epochs) if (int(value_epochs) > 0) else 1000

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
	except ValueError:
		print("Invalid input.")
		sys.exit(1)

def	main():
	W = []
	B = []

	LS_train = []
	AS_train = []

	X_train = np.array([[0.5, 3, 10],
						[2, 1.3, 6],
						[5, 1, 0.2]])
	Y_train = np.array([0, 1, 1])

	layers = [3] + hidden_layers + [2]
	print("Layers configuration:", layers)

	W = [
		np.array([[0.1, 0.4],
				[0.2, 0.5],
				[0.3, 0.6]]),

		np.array([[0.7, 0.9],
				[0.8, 1.0]]),

		np.array([[1.1, 1.3],
				[1.2, 1.4]])
	]

	B = [
			np.array([0, 0]),
			np.array([0, 0]),
			np.array([0, 0])
		]

	# print("this is w", W.shape())
	# print("this is b", B.shape())
	# sys.exit(1)

	for epoch in range(epochs):
		Z, A = feedForward(X_train, W, B)

		bce = loss(A[-1], Y_train)

		P_train = np.argmax(A[-1], axis=1)
		Ac_train = np.mean(P_train == Y_train)
		# print("this is Ac_train:", Ac_train)
		# sys.exit(1)

		print(
			f"epoch {epoch + 1}/{epochs} "
			f"- loss: {bce:.4f} "
			f"- accuracy: {Ac_train:.4f} "
		)

		AS_train.append(Ac_train)
		LS_train.append(bce)

		DW, DB = backPropagation(X_train, Y_train, W, Z, A)
		W, B = gradient_descent(W, B, DW, DB, n)

if __name__ == "__main__":
	main()