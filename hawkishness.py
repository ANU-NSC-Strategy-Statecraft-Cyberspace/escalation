import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def utility(c, r):
    if r == 1:
        return np.log(c+1)
    else:
        return ((c+1)**(1-r) - 1)/(1-r)
        
def inverse_utility(u, r):
    if r == 1:
        return np.exp(u) - 1
    else:
        return ((u * (1-r)) + 1)**(1/(1-r)) - 1

def expect(pWin, h, r):
    return inverse_utility(pWin * utility(2 + h, r) + (1-pWin) * utility(1 + h, r), r) - 1

def deal(pWin, hA, hB, rA, rB):
    wantA = expect(pWin, hA, rA)
    wantB = 1 - expect(1-pWin, hB, rB)
    return wantB - wantA

    
hMin = -1
hMax = 1
def diagram(pWin, r, ax):
    pWin = pWin
    rA = r
    rB = r
    hAs = np.linspace(hMin, hMax, 100)
    hBs = np.linspace(hMin, hMax, 100)
    hA, hB = np.meshgrid(hAs, hBs)
    im = ax.imshow(deal(pWin, hA, hB, rA, rB), cmap="RdYlGn", vmin=-1, vmax=1, extent=[hMin,hMax,hMin,hMax], origin='lower')
    plt.colorbar(im)
    return ax
    
def update_power(t, ax, fig, r, total):
    pWin = t/total
    fig.clear()
    ax = plt.gca()
    fig.suptitle('pWin: {}'.format(pWin))
    return diagram(pWin, r, ax)
    
def update_risk_aversion(t, ax, fig, pWin, total):
    r = (t/total)*2 - 1
    fig.clear()
    ax = plt.gca()
    fig.suptitle('Risk Aversion: {}'.format(r))
    return diagram(pWin, r, ax)
    
def video_power(r):
    fig, ax = plt.gcf(), plt.gca()
    ani = animation.FuncAnimation(fig, update_power, 101, fargs=(ax, fig, r, 100), interval=10, blit=False, repeat=False)
    plt.show()
    plt.close()
    
def video_risk_aversion(pWin):
    fig, ax = plt.gcf(), plt.gca()
    ani = animation.FuncAnimation(fig, update_risk_aversion, 101, fargs=(ax, fig, pWin, 100), interval=10, blit=False, repeat=False)
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    video_power(5)
    