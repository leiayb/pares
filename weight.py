import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

bf = 20.
rs = np.array([0.64, .67, .7, .72])
n = 14.
for r in rs:
    h = (n + 10**((bf + 78.4 + 97.7*np.log10(64))/165.2))/(1+r)
    print(h, r * h)
exit()

w = [129.9, 129., 129.8, 130.6, 131.4, 130.2, 129.4, 130.4, 129.4, 130., 128.8, 130., 130.6, 129.6, 131.2, 129.2,
     129.4, 129., 129.6, 131.2, 129., 129.4, 130.4, 130.4, 130.]
# w = w[5:]
t = range(len(w))

slope, intercept, r_value, p_value, std_err = stats.linregress(t, w)

plt.plot(t, w - np.average(w))
print(slope, slope * len(t), np.average(w))
y = slope * t + intercept
plt.plot(t, y - np.average(y))
# plt.show()

# plt.clf()
for i in t[1:]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(t[:i+1], w[:i+1])
    plt.scatter(t[i], slope)
plt.plot(t[1:], np.zeros(len(t[1:])))
plt.show()