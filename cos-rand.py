import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import pandas as pd
import matplotlib.animation as animation
from scipy.integrate import solve_ivp


def save_xls(mydict, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in mydict.items():
            pd.DataFrame(df).to_excel(writer, n, index=False)


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def mathieu(t, x, q, A):  # x = [x, dx/dt], returns dx/dt = [dx/dt, ddx/dtt]
    P = np.cos(2*t)
    # P = (np.sin(t))**10
    Q = A + 2 * q * P
    # M = np.array([[0, 1], [-Q, 0]])
    return [x[1], -Q * x[0]]


b = 20
grain = 20
As = np.linspace(-b, b, int(grain * b + 1))
Qs = np.linspace(-b, b, int(grain * b + 1))

cvar = 1.

ns = np.linspace(1, 20, 20)
ntot = roundup(np.log(2) / cvar)
hundred = np.linspace(25, ntot, int((ntot-20)/5))
ns = np.concatenate((ns, hundred))
print(cvar, np.log(2) / cvar, ntot)


def makemap(ns, dist):
    mus = {}
    empty_mu = np.zeros((np.size(As), np.size(Qs)))
    for n in ns:
        mus.update({str(int(n)): empty_mu.copy()})
    # musig = mus.copy()

    tmax = np.pi
    t = np.linspace(0, tmax, 1000)

    div = ns[-1]**3
    for k, A in enumerate(As):
        print(A)
        for j, Q in enumerate(Qs):
            Mtot = np.array([[1, 0], [0, 1]])
            for i, n in enumerate(ns):
                if i==0:
                    spacing = int(n)
                else:
                    spacing = int(ns[i] - ns[i-1])
                for _ in range(spacing):
                    if n==1:
                        q = Q
                    else:
                        if dist=='normal':
                            q = np.random.normal(Q, np.abs(Q) * cvar)
                        elif dist=='uniform':
                            q = np.random.uniform(-cvar * Q, cvar * Q)
                        else:
                            print('Distribution not valid')
                            exit()
                    
                    soln = solve_ivp(mathieu, [0, tmax], [1, 0], t_eval=t, args=(q, A))
                    xs1 = soln.y[0]
                    xd1 = soln.y[1]

                    soln = solve_ivp(mathieu, [0, tmax], [0, 1], t_eval=t, args=(q, A))
                    xs2 = soln.y[0]
                    xd2 = soln.y[1]

                    B = np.array([[xs1[-1], xs2[-1]], [xd1[-1], xd2[-1]]])
                    Mtot = np.matmul(B / div, Mtot)
                mu = np.log(np.max(np.abs(np.linalg.eig(Mtot)[0]))) / np.pi / n + np.log(div) / np.pi
                # mu = np.log(div * np.max(np.abs(np.real(np.linalg.eig(Mtot)[0])))) / np.pi / n
                # print(mu)
                mus[str(int(n))][k, j] = mu.copy()

    save_xls(mus, 'cos-maps/map-cos-'+dist+'.xlsx')
    # for i, n in enumerate(ns):
    #     # np.savetxt("cos-maps/map-cos-{}.csv".format(n), mus[i], delimiter=",")
    #     makeplot(n)
    
    # np.savetxt('cos-maps/map-cos-ALL.csv', mus, delimiter=',')


def makeplot(n):
    mus = pd.read_excel('cos-maps/map-cos-ALL.xlsx', None)
    # for n in ns:

    n = int(n)
    print(n)
    
    mun = mus[str(n)]
    plt.clf()
    plot = plt.pcolormesh(Qs, As, mun, cmap='inferno', shading='auto', alpha=1,
                            vmin=0, vmax=5)
    plot.set_edgecolor('face')
        # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
        # plt.clabel(cset, inline=True)
    plt.colorbar(plot)

    plt.xlabel(r'$\langle q \rangle$')
    plt.ylabel(r'$A$')
    # plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)=cos(2t)$")
    plt.title(r"$N={}$, $\sigma_q/\langle q \rangle=1/$".format(n)+str(int(cvar)))
    plt.tight_layout()
    # plt.show()
    plt.savefig('cos-maps/mathieu-stability-cos-rand-{}.pdf'.format(n))
    # plt.savefig('cos-maps/mathieu-stability-cos-rand-{}.png'.format(n))

# fig = plt.figure()


def makemovie(ns, dist):

    mus = pd.read_excel('cos-maps/map-cos-'+dist+'.xlsx', None)

    Writer = animation.writers['ffmpeg']
    fps = 2
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    fig = plt.figure()
    mu1 = mus[str(int(ns[0]))]
    plot = plt.pcolormesh(Qs, As, mu1, cmap='inferno',
                          shading='auto', vmin=0.0, vmax=5)
    # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black') 
    # plt.clabel(cset, inline=True)
    plt.colorbar(plot)

    plt.xlabel(r'$\langle q \rangle$')
    plt.ylabel(r'$A$')
    plt.tight_layout()
    plt.title('n = {} matrices from '.format(int(ns[-1]))+dist+' distribution')
    # plt.savefig(q+'-first-frame.png')
    # plt.xlim((bgs[0], bgs[-1]))
    # plt.ylim((effcs[0], effcs[-1]))
    with writer.saving(fig, "cos-maps/movie-cos"+dist+".mp4", 100):
        for n in ns:
            n = int(n)
            # muog = np.loadtxt('cos-maps/map-cos.csv', delimiter=',')
            mun = mus[str(n)]
            plot = plt.pcolormesh(Qs, As, mun, cmap='inferno',
                                  shading='auto', vmin=0, vmax=5)
            # cset = plt.contour(B, E, p[j, 1:, 1:], colors='black')
            # plt.clabel(cset, inline=True)
            # plt.colorbar(plot)
            plt.xlabel(r'$\langle q \rangle$')
            plt.ylabel(r'$A$')
            plt.tight_layout()
            plt.title(r"$N={}$, $\sigma_q/\langle q \rangle=1$".format(int(n)))
            writer.grab_frame()
            # plt.show()


if __name__=='__main__':
    distrs = ['uniform', 'normal']
    for dist in distrs:
        makemap(ns, dist)
    # for n in ns:
    #     makeplot(n)
        makemovie(ns, dist)