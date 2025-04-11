from typing import Callable
import numpy as np
import random
from matplotlib import pyplot as plt


class Smoothing:
    """
    Класс для реализации сглаживания

    :param func тригонометрическая функция для создания сигналов
    """

    @staticmethod
    def make_some_noise(n: int = 500):
        return np.random.normal(0, 0.5, size=n)

    def __init__(self, func: Callable[[float], float], n: int = 500):
        self.func = func
        self.x = np.linspace(0, 1, n)
        self.noise = self.make_some_noise(n=n)
        self.y = [func(x) for x in self.x] + self.noise

    def make_blowout(self):
        b = random.randint(20, 50)
        self.y[random.randint(0, len(self.y))] = b

    def moving_average(self, k: int = 10) -> list[float]:
        n = len(self.x)
        ys_vals = []

        for s in range(k, n):
            window = self.y[s - k:s]
            avg = sum(window) / k
            ys_vals.append(avg)

        return ys_vals

    def exponential_moving_average(self, alpha: float = 0.001) -> list[float]:
        y = [alpha * float(self.x[i]) + (1 - alpha) * self.y[i-1] for i in range(1, len(self.x))]
        return y
    def moving_median(self, k: int = 10) -> list[float]:
        y = [float(np.median(self.y[i:i + k])) for i in range(len(self.x) - k)]
        return y

    def plot_original_signal(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, label="Исходный сигнал", color="blue")
        plt.title("Исходный сигнал")
        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_moving_average(self, k: int = 10, with_original_signal: bool = False):
        plt.figure(figsize=(10, 6))
        y = self.moving_average(k=k)
        x = self.x[:-k]
        plt.plot(x, y, label="Сглаженный сигнал скользящим средним", color="red")
        if with_original_signal:
            plt.plot(self.x[:-k], self.y[:-k], label="Исходный сигнал", color="blue")
        plt.title("Сглаженный сигнал скользящим средним")
        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_moving_median(self, k: int = 10, with_original_signal: bool = False):
        plt.figure(figsize=(10, 6))
        y = self.moving_median(k=k)
        x = self.x[:-k]
        plt.plot(x, y, label="Сглаженный сигнал скользящей медианой", color="red")
        if with_original_signal:
            plt.plot(self.x[:-k], self.y[:-k], label="Исходный сигнал", color="blue")
        plt.title("Сглаженный сигнал скользящей медианой")
        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_exponential_moving_average(self, alpha: float = 0.001, with_original_signal: bool = False):
        plt.figure(figsize=(10, 6))
        y = self.exponential_moving_average(alpha=alpha)
        k = len(self.y) - len(y)
        x = self.x[:-k]
        plt.plot(x, y, label="Экспоненциальное сглаживание сигнала", color="red")
        if with_original_signal:
            plt.plot(self.x[:-k], self.y[:-k], label="Исходный сигнал", color="blue")
        plt.title("Экспоненциальное сглаживание сигнала")
        plt.xlabel("t")
        plt.ylabel("f(t)")
        plt.legend()
        plt.grid(True)
        plt.show()
