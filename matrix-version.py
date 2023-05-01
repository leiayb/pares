import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

func = 'cos'
if func == 'sin10':
    fstr = '\sin^{10}(t)'
if func == 'cos':
    fstr = '\cos(2t)'

def mathieu(t, x, q, A):  # x = [x, dx/dt], returns dx/dt = [dx/dt, ddx/dtt]
    P = np.cos(2*t)
    # P = (np.sin(t))**10
    Q = A + 2 * q * P
    # M = np.array([[0, 1], [-Q, 0]])
    return [x[1], -Q * x[0]]


# n = 50  # number of periods

# qs = np.linspace(-3, 3, 16)
# As = np.linspace(15, 17, 11)

# As = [-5, 5]
# qs = [2]
b = 20
grain = 20
As = np.linspace(-b, b, int(grain * b + 1))
qs = np.linspace(-b, b, int(grain * b + 1))

mus = np.zeros((np.size(As), np.size(qs)))

plt.figure()

for j, q in enumerate(qs):
    for k, A in enumerate(As):
        tmax = np.pi
        t = np.linspace(0, tmax, 1000)
        
        soln = solve_ivp(mathieu, [0, tmax], [1, 0], t_eval=t, args=(q, A))
        xs1 = soln.y[0]
        xd1 = soln.y[1]

        soln = solve_ivp(mathieu, [0, tmax], [0, 1], t_eval=t, args=(q, A))
        xs2 = soln.y[0]
        xd2 = soln.y[1]

        # numerically evolved solution
        # xi = xs1 + xs2
        # ti = t[xi > 0]
        # xi = xi[xi > 0]

        # matrix evolved solution
        # xsm = [[1, 1]]
        # tsm = [0]
        # tp = t[t <= np.pi][-1]
        B = np.array([[xs1[-1], xs2[-1]],
                      [xd1[-1], xd2[-1]]])
        # print(B, np.linalg.det(B), np.log(np.linalg.eig(B)[0][-1]) / np.pi)
        mu = np.log(np.max(np.abs(np.linalg.eig(B)[0]))) / np.pi
        # print(B, xsm[-1])
        # for i in range(n):
        #     tsm.append(np.pi * (i+1))
        #     xsm.append(B @ xsm[-1])

        # plt.figure()
        # plt.semilogy(t, xi, label='full sol')
        # plt.semilogy(t, xs1, label='x1', alpha=0.5)
        # plt.semilogy(t, xs2, ':', label='x2')

        # # print(xsm)
        # # exit()
        # plt.semilogy(tsm, np.array(xsm)[:, 0], '--', label='matrix sol')
        # plt.title(str(A)+' '+str(q))
        # plt.legend()
        # plt.show()

        if func == 'sin10':
            mus[k, j] = np.log(np.abs(mu))
        if func == 'cos':
            mus[k, j] = np.abs(mu)

np.savetxt("map-{}-eig.csv".format(func), mus, delimiter=",")

mus = np.loadtxt('map-{}-eig.csv'.format(func), delimiter=',')

plot = plt.pcolormesh(qs, As, mus, cmap='inferno', shading='auto', alpha=1)
plot.set_edgecolor('face')
    # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
    # plt.clabel(cset, inline=True)
plt.colorbar(plot)

plt.xlabel(r'$q$')
plt.ylabel(r'$A$')
plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)={}$".format(fstr))
plt.tight_layout()
# plt.show()
plt.savefig('mathieu-stability-{}-eig.pdf'.format(func))
# plt.savefig('mathieu-stability-{}-eig.png'.format(func))


# interp = RegularGridInterpolator((qs, As), mus)

# input A, q for some reason
