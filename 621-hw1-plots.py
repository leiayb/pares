import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['crimson', 'darkorange', 'gold', 'limegreen', 'royalblue']

x = np.linspace(-3, 3)

def f(k, h):
    return -np.log(np.exp(k) * np.cosh(h) + np.sqrt(np.exp(2*k) * (np.sinh(h))**2 + np.exp(-2*k)))

def m(k, h):
    return np.exp(k) * np.sinh(h) / np.sqrt(np.exp(2*k) * (np.sinh(h))**2 + np.exp(-2*k))

J = 1

plt.figure()
for i, T in enumerate(np.linspace(0.5, 4, 5)):
    plt.plot(x*T, f(J/T, x), label=str(T)+', '+str(round(J/T, 3)), color=colors[i])
    plt.title(r'coupling $J=1$')
    plt.xlim([-3, 3])
    plt.xlabel(r'external field $H$')
    plt.ylabel(r'free energy per site $f$')
    plt.legend(title=r'$k_BT,\ K$')
plt.tight_layout()
plt.savefig('f-true.pdf')

plt.figure()
for i, T in enumerate(np.linspace(0.5, 4, 5)):
    plt.plot(x, f(J/T, x), label=str(T)+', '+str(round(J/T, 3)), color=colors[i])
    plt.xlim([-3, 3])
    plt.title(r'coupling $J=1$')
    plt.xlabel(r'dimensionless external field $h=H/T$')
    plt.ylabel(r'free energy per site $f$')
    plt.legend(title=r'$k_BT, K$')
plt.tight_layout()
plt.savefig('f-h.pdf')

plt.figure()
for i, T in enumerate(np.linspace(0.5, 4, 5)):
    plt.plot(x*T, m(J/T, x), label=str(T)+', '+str(round(J/T, 3)), color=colors[i])
    plt.title(r'coupling $J=1$')
    plt.xlim([-3, 3])
    plt.xlabel(r'external field $H$')
    plt.ylabel(r'magnetization per site $M$')
    plt.legend(title=r'$k_BT,\ K$')
plt.tight_layout()
plt.savefig('m-true.pdf')

plt.figure()
for i, T in enumerate(np.linspace(0.5, 4, 5)):
    plt.plot(x, m(J/T, x), label=str(T)+', '+str(round(J/T, 3)), color=colors[i])
    plt.xlim([-3, 3])
    plt.title(r'coupling $J=1$')
    plt.xlabel(r'dimensionless external field $h=H/T$')
    plt.ylabel(r'magnetization per site $M$')
    plt.legend(title=r'$k_BT, K$')
plt.tight_layout()
plt.savefig('m-h.pdf')