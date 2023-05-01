import numpy as np
import matplotlib.pyplot as plt
import cmath

b = 20
grain = 20
As = np.linspace(-b, b, int(grain * b + 1))
# As = np.linspace(0, 10, int(grain * b + 1))
qs = np.linspace(-b, b, int(grain * b + 1))

mus = np.zeros((np.size(As), np.size(qs)))
ans = mus.copy()

for j, q in enumerate(qs):
    for k, A in enumerate(As):
        # print(q, A)
        if np.abs(A) > 0:
            p = np.pi * cmath.sqrt(A)
            x1 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A)
            x2 = q * (cmath.cos(p) - 1) / A + cmath.sin(p) / cmath.sqrt(A)
            dx1 = -q * (1 + cmath.cos(p)) - cmath.sqrt(A) * cmath.sin(p)
            dx2 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A)
            M = [[x1, x2], [dx1, dx2]]
            # print(M, A, q, p)
            # exit()
            # analytic1 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A) + cmath.sqrt(A*(q**2-A) * cmath.sin(p)**2 - A * cmath.sqrt(A) * q * cmath.sin(2*p)) / A
            # analytic2 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A) - cmath.sqrt(A*(q**2-A) * cmath.sin(p)**2 - A * cmath.sqrt(A) * q * cmath.sin(2*p)) / A
            analytic1 = x1 + cmath.sqrt(x1**2 - 1)
            analytic2 = x1 - cmath.sqrt(x1**2 - 1)
        else:
            M = [[1 - q * np.pi, np.pi * (1 - q * np.pi / 2)],
                 [-2 * q, 1 - q * np.pi]]
            qp = q * np.pi
            analytic1 = 1 - qp + cmath.sqrt(qp) * cmath.sqrt(qp - 2)
            analytic2 = 1 - qp - cmath.sqrt(qp) * cmath.sqrt(qp - 2)
            # print(np.linalg.det(M), np.linalg.eig(M)[0], [analytic1, analytic2])
            # exit()
        # mu = np.log(np.max(np.abs(np.real(np.linalg.eig(M)[0])))) / np.pi
        mu = np.log(np.max(np.abs(np.linalg.eig(M)[0]))) / np.pi
        # ans[k, j] = np.log(np.max(np.abs(np.real([analytic1, analytic2])))) / np.pi
        ans[k, j] = np.log(np.max(np.abs([analytic1, analytic2]))) / np.pi
        mus[k, j] = mu

np.savetxt("delta-maps/map-delta-analytic.csv", ans, delimiter=",")
np.savetxt("delta-maps/map-delta.csv", mus, delimiter=",")

mus = np.loadtxt('delta-maps/map-delta.csv', delimiter=',')

plt.clf()
qplot = 1.25 / 2
# print(qs[np.argwhere(qs>qplot)[0][0]])
muq = mus[:, np.argwhere(qs>qplot)[0][0]]

plt.clf()
plt.semilogy(As, muq * np.pi)
plt.ylim([.0001, 10])
plt.xlim([0, 10])
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\gamma$')
plt.title(r'$q=1.25$')
# plt.show()
plt.savefig('q125.pdf')


plt.clf()
plot = plt.pcolormesh(qs, As, mus, cmap='inferno', shading='auto', alpha=1)
plot.set_edgecolor('face')
    # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
    # plt.clabel(cset, inline=True)
plt.colorbar(plot)

plt.xlabel(r'$q$')
plt.ylabel(r'$A$')
plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)=\delta(t-\frac{\pi}{2}n)$")
plt.tight_layout()
# plt.show()
plt.savefig('mathieu-stability-delta.pdf')
# plt.savefig('mathieu-stability-delta.png')

ans = np.loadtxt('delta-maps/map-delta-analytic.csv', delimiter=',')

plt.clf()
plot = plt.pcolormesh(qs, As, ans, cmap='inferno', shading='auto', alpha=1)
plot.set_edgecolor('face')
    # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
    # plt.clabel(cset, inline=True)
plt.colorbar(plot)

plt.xlabel(r'$q$')
plt.ylabel(r'$A$')
plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)=\delta(t-\frac{\pi}{2}n)$")
plt.tight_layout()
# plt.show()
plt.savefig('mathieu-stability-delta-analytic.pdf')

# plt.clf()
# plot = plt.pcolormesh(qs, As, mus - ans, cmap='inferno', shading='auto', alpha=1)
# plot.set_edgecolor('face')
#     # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
#     # plt.clabel(cset, inline=True)
# plt.colorbar(plot)

# plt.xlabel(r'$q$')
# plt.ylabel(r'$A$')
# plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)=\delta(t-\frac{\pi}{2}n)$")
# plt.tight_layout()
# # plt.show()
# plt.savefig('mathieu-stability-delta-diff.pdf')