from smoothing import Smoothing
from math import sin

func = lambda t: sin(4 * t ** 3)
s = Smoothing(func=func)
s.make_blowout()

s.plot_moving_average(k=5, with_original_signal=True)
s.plot_moving_median(k=4, with_original_signal=True)
s.plot_exponential_moving_average(alpha=0.15, with_original_signal=True)
