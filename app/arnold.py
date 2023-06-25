import cv2
import numpy as np

import matplotlib.pyplot as plt

def matrix_power_mod(matrix: np.ndarray, power: int, mod: np.ndarray):
	result = np.eye(matrix.shape[0], dtype=np.int64)
	for _ in range(power):
		result = np.matmul(result, matrix) % mod
	return result

def arnold(img: np.ndarray, n: int):
	img = np.array(img)
	h, w, c = img.shape
	new_img = np.zeros((h, w, c), dtype=np.uint8)
	matrix = np.array([[2, 1], [1, 1]])
	N_matrix = matrix_power_mod(matrix, n, np.array([h, w]))
	N_matrix = N_matrix % np.array([h, w])
	for x in range(h):
		for y in range(w):
			new_x, new_y = np.dot(N_matrix, np.array([x, y])) % np.array([h, w])
			new_img[new_x, new_y] = img[x, y]
	return new_img

def get_arnold_reverse(L: int):
	_matrix = np.array([[2, 1], [1, 1]])
	matrix = _matrix.copy()
	cnt = 1
	while matrix[0, 0] != 1 or matrix[0, 1] != 0 or matrix[1, 0] != 0 or matrix[1, 1] != 1:
		matrix = np.matmul(matrix, _matrix) % np.array([L, L])
		cnt += 1
	return cnt

def arnold_reverse(img: np.ndarray, n: int):
	nn = get_arnold_reverse(img.shape[0])
	reversed_img = arnold(img, nn - n)
	return reversed_img