import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import matplotlib.animation as animation
import pandas as pd

dirname = 'delta-maps/'


def save_xls(mydict, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in mydict.items():
            pd.DataFrame(df).to_excel(writer, n, index=False)


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100.


b = 20.
grain = 2
As = np.linspace(-b, b, int(grain * b + 1))
Qs = np.linspace(-b, b, int(grain * b + 1))


def makemap(cvar, dist):

    ns = np.linspace(1, 20, 20)
    if cvar != 0:
        ntot = roundup(np.log(2) / cvar)
        hundred = np.linspace(25, ntot, int((ntot-20)/5))
        ns = np.concatenate((ns, hundred))
        print(cvar, np.log(2) / cvar, ntot)

    mus = {}
    mus.update({'ns': ns})
    empty_mu = np.zeros((np.size(As), np.size(Qs)))
    for n in ns:
        mus.update({str(int(n)): empty_mu.copy()})

    div = ns[-1]**3
    # div = 10.0
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
                    if n==1 or cvar==0:
                        q = Q
                    else:
                        if dist=='normal':
                            q = np.random.normal(Q, np.abs(Q) * cvar)
                        elif dist=='uniform':
                            q = np.random.uniform(Q - cvar * Q, Q + cvar * Q)  # [x] HUGE bug fixed here (+Q's)
                        else:
                            print('Distribution not valid')
                            exit()

                    if np.abs(A) > 0:
                        p = np.pi * cmath.sqrt(A)
                        x1 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A)
                        x2 = q * (cmath.cos(p) - 1) / A + cmath.sin(p) / cmath.sqrt(A)
                        dx1 = -q * (1 + cmath.cos(p)) - cmath.sqrt(A) * cmath.sin(p)
                        dx2 = cmath.cos(p) - q * cmath.sin(p) / cmath.sqrt(A)
                        M = [[x1, x2], [dx1, dx2]]
                    else:
                        M = [[1 - q * np.pi, np.pi * (1 - q * np.pi / 2)],
                            [-2 * q, 1 - q * np.pi]]
                    M = np.array(M)
                    # print(M, Mtot)
                    Mtot = np.matmul(M / div, Mtot)

                mu = np.log(np.max(np.abs(np.linalg.eig(Mtot)[0]))) / n / np.pi + np.log(div) / np.pi
                mus[str(int(n))][k, j] = mu.copy()

    # np.savetxt("delta-maps/map-delta-{}.csv".format(n), mus, delimiter=",")
    save_xls(mus, dirname+'map-delta-'+dist+'-{}.xlsx'.format(str(cvar)))
    

def steady_states(cvar, dist):

    print(cvar)

    mus = pd.read_excel(dirname+'map-delta-'+dist+'-{}.xlsx'.format(str(cvar)), None)
    for k, _ in mus.items():
        mus[k] = np.squeeze(mus[k].to_numpy())

    zs = []
    for n in mus['ns']:
        zs.append(np.std(mus['1'] - mus[str(int(n))]))  # [x] inf's mess up std
    # plt.clf()
    plt.semilogy(mus['ns'], zs, label=str(cvar))
    plt.xlabel('Number of random matrices')
    plt.title(dist)
    plt.ylabel(r'$\langle |\Delta \mu| \rangle$')
    # plt.ylim([0, 1.8])
    plt.legend(title=r'$q/\sigma$')
    # plt.title(r'$\sigma_q = q/{}$'.format(cvar))
    # plt.savefig(dirname+'steady-state-{}.pdf'.format(cvar))


def makeplot(n, cvar):
    mus = pd.read_excel(dirname+'map-delta-ALL-{}.xlsx'.format(str(cvar)), None)
    # for n in ns:

    n = int(n)
    print(n)
    
    mun = mus[str(n)]
    plt.clf()
    plot = plt.pcolormesh(Qs, As, mun, cmap='inferno', shading='auto', alpha=1,
                        #   vmin=-5, vmax=5
                          )
    plot.set_edgecolor('face')
        # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black')
        # plt.clabel(cset, inline=True)
    plt.colorbar(plot)

    plt.xlabel(r'$\langle q \rangle$')
    plt.ylabel(r'$A$')
    # plt.title(r"|Re($\mu$)| of Mathieu's equation solution with $P(t)=cos(2t)$")
    plt.title(r"$N={}$, $\sigma_q/\langle q \rangle=$".format(n)+str(cvar))
    plt.tight_layout()
    # plt.show()
    plt.savefig(dirname+'mathieu-stability-delta-rand-{}.pdf'.format(cvar))
    # plt.savefig('cos-maps/mathieu-stability-cos-rand-{}.png'.format(n))


def makemovie(cvar, dist):
    mus = pd.read_excel(dirname+'map-delta-'+dist+'-{}.xlsx'.format(str(cvar)), None)
    ns = np.squeeze(mus['ns'].to_numpy())

    Writer = animation.writers['ffmpeg']
    fps = 2
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    fig = plt.figure()
    mu1 = mus[str(int(ns[0]))]
    plot = plt.pcolormesh(Qs, As, mu1, cmap='inferno',
                          shading='auto', vmin=0, vmax=5)
    # cset = plt.contour(B, E, p[0, 1:, 1:], colors='black') 
    # plt.clabel(cset, inline=True)
    plt.colorbar(plot)

    plt.xlabel(r'$\langle q \rangle$')
    plt.ylabel(r'$A$')
    plt.tight_layout()
    plt.title('n = {} matrices'.format(int(ns[-1])))
    # plt.savefig(q+'-first-frame.png')
    # plt.xlim((bgs[0], bgs[-1]))
    # plt.ylim((effcs[0], effcs[-1]))
    with writer.saving(fig, "delta-maps/movie-delta-"+dist+"-{}.mp4".format(cvar), 100):
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
            plt.title(r"$N={}$, $\sigma_q/\langle q \rangle=$".format(n)+str(cvar))
            writer.grab_frame()
            # plt.show()


def compare(ell, dist):

    data = pd.read_excel(dirname+'map-delta-'+dist+'-1.0.xlsx', None)
    mus = data[str(int(np.squeeze(data['ns'].to_numpy())[-1]))]

    qplot = 5. / (2**ell)
    # print(qs[np.argwhere(qs>qplot)[0][0]])
    mus = np.squeeze(mus.to_numpy())
    muq = mus[:, np.argwhere(Qs>qplot)[0][0]]
    print(muq)

    plt.semilogy(As, muq * np.pi, label=ell)
    # plt.show()


if __name__=='__main__':

    distrs = ['uniform', 'normal']
    # distrs = ['uniform']
    # cvars = [.001, .01, 0.1, 1., 10., 100.]
    cvars = [1.]
    # cvars = [0]  # sanity check

    """ makemaps """

    for dist in distrs:
        for cvar in cvars:
            makemap(cvar, dist)

    """ average deviation (steady states) """

    # for dist in distrs:
    #     plt.clf()
    #     for cvar in cvars:
    #         steady_states(cvar, dist)
    #     plt.savefig(dirname+'steady-states-'+dist+'.pdf')

    """ individual plots """

        # for n in ns:
        #     makeplot(n, cvar)

    """ movies """

    # for dist in distrs:
    #     for cvar in cvars:
    #         makemovie(cvar, dist)

    """ mu vs specific q's """

    # for dist in distrs:
    #     plt.clf()
    #     for ell in [4, 5, 6, 7, 8]:
    #         print(ell)
    #         compare(ell, dist)
    #     # plt.ylim([1e-6, 1])
    #     # plt.xlim([0, 10])
    #     plt.xlabel(r'$\lambda$')
    #     plt.ylabel(r'$\gamma$')
    #     if dist=='uniform':
    #         plt.title('Uniform [-q, q]')
    #     if dist=='normal':
    #         plt.title(r'$\sigma/q=1$')
    #     plt.legend(title=r'$\langle q\rangle=10/2^\ell$')
    #     plt.savefig(dirname+'qs-'+dist+'.pdf')