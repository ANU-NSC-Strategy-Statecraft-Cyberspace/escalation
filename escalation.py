from random import random, choice, sample
from statistics import median
from itertools import combinations
import csv
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
from functools import lru_cache as memoize
import colorsys

"""
Important ideas:

Want to find dynamics where the population's genes significantly affect the implicit fitness function
 Consider moves A, B.
 Then: (AA, AB, BA, BB):
 (0,0,0,0) - no evolution
 (0,0,0,1) - evolve B
 (0,0,1,0) - evolve unhappy B's
 (0,0,1,1) - evolve B
 (0,1,0,0) - evolve unhappy A's
 (0,1,0,1) - stable 50/50 split
 (0,1,1,0) - stable 50/50 split
 (0,1,1,1) - evolve B
 (1,0,0,0) - evolve A
 (1,0,0,1) - symmetry break into either all A or all B
 (1,0,1,0) - stable 50/50 split
 (1,0,1,1) - unstable at A, stable at B
 (1,1,0,0) - evolve A
 (1,1,0,1) - unstable at B, stable at A
 (1,1,1,0) - evolve A
 (1,1,1,1) - no evolution

Consider different war costs
Plot evolved hawkishness against war costs
Consider "evolution over time" graph with some shocks where war cost changes partway through


"""

loot = 1
loss = 1

mutate = 0.05
replace = 5
population = 100
evolve_steps = 1000

peace_limit = 4
turn_limit = 100


#assert loss == 1
#@memoize(maxsize=16384)
#def expect_value(pA, pB):
#    return ( ((pA * loot) - (pB)) / (pA + pB), ((pB * loot) - (pA)) / (pA + pB) )

@memoize(maxsize=16384)
def p_win(pA, pB):
    return ( pA / (pA + pB), pB / (pA + pB) )

basecyber = {'rich': 10, 'medium': 1, 'poor': 0.1}
basepower = {'rich': 100, 'medium': 10, 'poor': 1}
sortable = {'rich': 3, 'medium': 2, 'poor': 1}

@memoize()
def compare_inner(p1, p2):
    if sortable[p1] > sortable[p2]:
        return 'g'
    elif sortable[p1] < sortable[p2]:
        return 'l'
    else:
        return 'e'

@memoize()
def compare(c1, c2, p1, p2):
    return (compare_inner(c1, c2), compare_inner(c2, c1), compare_inner(p1, p2), compare_inner(p2, p1))

def newhawk():
    return random()

def newrisk():
    return random()

class Gene:
    def __init__(self):
        self.hawk = newhawk()
        self.risk = newrisk()

    def force(self, hawk, risk):
        self.hawk = hawk
        self.risk = risk

    def inherit(self, parent1, parent2):
        if random() < mutate:
            self.hawk = newhawk()
        else:
            self.hawk = choice([parent1.hawk, parent2.hawk])

        if random() < mutate:
            self.risk = newrisk()
        else:
            self.risk = choice([parent1.risk, parent2.risk])


genechoices = ['g', 'e', 'l']

#@memoize(maxsize=16384)
def genesget(state, other):
    c1, c2, p1, p2 = compare(state.cyber, other.cyber, state.power, other.power)
    return (state.genes.chromosomes[c1][p1], other.genes.chromosomes[c2][p2])

class Genes:
    def __init__(self):
        self.chromosomes = {c: {p: Gene() for p in genechoices} for c in genechoices}

    def force(self, hawk, risk):
        for genes in self.chromosomes.values():
            for gene in genes.values():
                gene.force(hawk, risk)

    def inherit(self, parent1, parent2):
        for c, genes in self.chromosomes.items():
            for p, gene in genes.items():
                gene.inherit(parent1.chromosomes[c][p], parent2.chromosomes[c][p])


class State:
    def __init__(self, cyber, power):
        self.cyber = cyber
        self.power = power
        self.genes = Genes()
        self.reset()

    def force(self, hawk, risk):
        self.genes.force(hawk, risk)
        self.reset()

    def reset(self):
        self.currentcyber = basecyber[self.cyber]
        self.currentpower = basepower[self.power]

    def choice(self, other):
        P_WIN, P_opp_WIN = p_win(self.currentpower, other.currentpower)
        gene, othergene = genesget(self, other)
        if P_WIN > gene.risk:
            if self.currentcyber <= other.currentcyber and random() < gene.hawk:
                return "KINETIC"
            else:
                return "CYBER"
        elif P_opp_WIN > othergene.risk:
            if self.currentcyber <= other.currentcyber and random() < gene.hawk:
                return "KINETIC"
            else:
                return "CYBER"
        else:
            if random() < gene.hawk:
                return "CYBER"
            else:
                return "PEACE"

assert loss == 1
def payoff(result, A, B):
    if result == "WAR":
        if random() * (A.currentpower + B.currentpower) < (A.currentpower):
            A.score += loot
            B.score -= 1
        else:
            A.score -= 1
            B.score += loot

def calcstats(round, states):
    return [median([s.genes.chromosomes[c][p].hawk for s in states]) for c in genechoices for p in genechoices] + [median([s.genes.chromosomes[c][p].risk for s in states]) for c in genechoices for p in genechoices]

def opencsv(f):
    writer = csv.writer(f)
    writer.writerow(['h' + c + p for c in genechoices for p in genechoices] + ['r' + c + p for c in genechoices for p in genechoices])
    return writer

def writetocsv(c, genestats):
    c.writerow(genestats)

def evolve():
    print("BEGIN!")
    global loot
    with open('evolve.csv', 'w') as f:
        c = opencsv(f)
        totalstats = []
        #states = [State(c, p) for _ in range(population//9) for c in basecyber for p in basepower]
        states = [State('medium', 'medium') for _ in range(population)]
        for round in range(evolve_steps):
            #if round == evolve_steps//2:
                #expect_value.cache_clear()
                #p_win.cache_clear()
                #loot = 1
            for s in states:
                s.score = 0
            for turn in range(1):
                for A, B in combinations(states, 2):
                    A.reset()
                    B.reset()
                    payoff(eval1(A,B), A, B)

                    A.reset()
                    B.reset()
                    payoff(eval1(B,A), B, A)
            states.sort(key=lambda s: s.score)
            removed = sample(states[:population//2], replace)
            for s in removed:
                s.genes.inherit(choice(states[population//2:]).genes, choice(states[population//2:]).genes)
            genestats = calcstats(round, states)
            writetocsv(c, genestats)
            print(round)

def graphevolve():
    X = np.loadtxt('evolve.csv', delimiter=',', skiprows=1)
    plt.rc('axes', prop_cycle=(cycler('color', ['red', 'orange', 'yellow', 'green', 'turquoise', 'blue', 'purple', 'brown', 'black']*2) + cycler('linestyle', ['-']*9 + ['--']*9)))
    plt.plot(X)
    plt.legend(['-']*4 + ['hawk'] + ['-']*8 + ['risk'] + ['-']*4)
    plt.show()


def graphevolvetwo(X, t):
    fig, ax = plt.subplots()
    ax.set_xlabel('Time', size=14)
    ax.text(-0.02,0,'Low', size=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.text(-0.02,1,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.tick_params(bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
    lines = plt.plot(X, linewidth=3)
    plt.setp(lines[0], color='b')
    plt.setp(lines[1], color='limegreen')
    plt.xlim(t, t+evolve_steps)
    plt.legend(['Hawkishness'] + ['Risk Aversion'], loc=(-0.28, 0.5))
    return fig, ax, lines[0], lines[1]

def graphevolvebase():
    plt.rc('figure', figsize=(10,5))
    X = np.loadtxt('evolve.csv', delimiter=',', skiprows=1)
    X=X[:,[4,13]]
    graphevolvetwo(X, 0)
    plt.subplots_adjust(left=0.22, bottom=0.06, right=0.98, top=0.98)
    plt.show()


def graphevolvebaseanim():
    plt.rc('figure', figsize=(10,5))
    X = np.loadtxt('evolve.csv', delimiter=',', skiprows=1)
    X=X[:,[4,13]]
    t = -evolve_steps
    fig, ax, _, _ = graphevolvetwo(X, t)
    def update_line(num, t, ax):
        num *= 2
        if t+num <= 0:
            ax.set_xlim(t+num, t+num+evolve_steps)
        else:
            ax.set_xlim(0, evolve_steps)
        return ax,
    line_ani = animation.FuncAnimation(fig, update_line, evolve_steps, fargs=(t, ax),
                                       interval=5, blit=False)
    plt.subplots_adjust(left=0.22, bottom=0.06, right=0.98, top=0.98)
    #plt.show()
    line_ani.save('anim.mp4', fps=60, bitrate=1000)

def graphevolvebasegrow():
    plt.rc('figure', figsize=(10,5))
    X = np.loadtxt('evolve.csv', delimiter=',', skiprows=1)
    X=X[:,[4,13]]
    t = -evolve_steps
    fig, ax, line1, line2 = graphevolvetwo(X, t)
    def update_line(num, t, ax, line1, line2):
        num *= 2
        ax.set_xlim(0, evolve_steps)
        if num <= evolve_steps:
            line1.set_data(np.arange(0,num,1), X[:num,0])
            line2.set_data(np.arange(0,num,1), X[:num,1])
        else:
            line1.set_ydata(X[:,0])
            line2.set_ydata(X[:,1])
        return ax, line1, line2
    line_ani = animation.FuncAnimation(fig, update_line, evolve_steps, fargs=(t, ax, line1, line2),
                                       interval=5, blit=False)
    plt.subplots_adjust(left=0.22, bottom=0.06, right=0.98, top=0.98)
    #plt.show()
    line_ani.save('grow.mp4', fps=60, bitrate=1000)

def eval1(A, B):
    peace_count = 0
    for _ in range(turn_limit):
        choice = A.choice(B)
        if choice == "CYBER":
            A.currentpower += A.currentcyber
            peace_count = 0
        elif choice == "PEACE":
            peace_count += 1
        else:
            return "WAR"

        if peace_count >= peace_limit:
            return "PEACE"

        choice = B.choice(A)
        if choice == "CYBER":
            B.currentpower += B.currentcyber
            peace_count = 0
        elif choice == "PEACE":
            peace_count += 1
        else:
            return "WAR"

        if peace_count >= peace_limit:
            return "PEACE"

    return "STANDOFF"

def twoplayersinglerun(hawk1, hawk2, risk1, risk2):
    A = State('medium', 'medium')
    B = State('medium', 'medium')
    A.force(hawk1, risk1)
    B.force(hawk2, risk2)
    war = 0
    not_peace = 0
    steps = 100
    for _ in range(steps):
        A.reset()
        B.reset()
        result = eval1(A, B)
        if result == 'WAR':
            war += 1
        if result != 'PEACE':
            not_peace += 1
        result = eval1(B, A)
        if result == 'WAR':
            war += 1
        if result != 'PEACE':
            not_peace += 1
    print(hawk1, hawk2, risk1, risk2, war/(steps*2), not_peace/(steps*2))
    return (war/(steps*2), not_peace/(steps*2))

hprec = 0.02
rprec = 0.02
hnum = round(1/hprec)
rnum = round(0.52/rprec)


def twoplayeranim():
    def update_line(num, data, ax):
        ax.imshow(data[..., num])
        return ax,
    hawk1 = 1
    hawk2 = 1
    X = np.array([[hawk1, hawk2, risk1, risk2, *twoplayersinglerun(hawk1, hawk2, risk1, risk2)] for risk1 in np.arange(0.5 - rprec,1.0 + rprec,rprec) for risk2 in np.arange(0.5 - rprec,1.0 + rprec,rprec)])
    np.savetxt('data.csv', X, fmt='%.3f', delimiter=',')

def stillfile():
    X = np.loadtxt('data.csv', delimiter=',')
    data = np.zeros((rnum+1,rnum+1, 3))
    for hawk1, hawk2, risk1, risk2, war, not_peace in X:
        #color = (not_peace, not_peace - war, 1.0 - not_peace)
        color = (not_peace, 1.0 - war, 0)
        print(int(round(hawk1*hnum)), int(round((risk1-0.48)*rnum/0.52)))
        data[int(round((risk1-0.48)*rnum/0.52)), int(round((risk2-0.48)*rnum/0.52))] = color

    fig, ax = plt.subplots()
    ax.set_xlabel('Country 1 Risk Aversion', size=14)
    ax.set_ylabel('Country 2 Risk Aversion', size=14)
    ax.text(-0.02,0,'Low', size=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.text(-0.02,1,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0,-0.02,'Low', size=12, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.text(1,-0.02,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.tick_params(bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
    plt.legend(handles=[mpatches.Patch(color='red', label='War'), mpatches.Patch(color='gold', label='Standoff'), mpatches.Patch(color='limegreen', label='Peace')], loc=(1.05, 0.8))
    ax.imshow(data, origin='lower', extent=(0.0, 1.0, 0.0, 1.0))
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.76, top=0.98)
    plt.show()

def animfileforconversation():
    X = np.loadtxt('data.csv', delimiter=',')
    print(np.shape(X))
    data = np.zeros((hnum+1,rnum+1,rnum+1, 3))
    for hawk1, hawk2, risk1, risk2, war, not_peace in X:
        #color = (not_peace, not_peace - war, 1.0 - not_peace)
        color = (not_peace, 1.0 - war, 0)
        data[int(round(hawk1*hnum)), int(round((risk1-0.48)*rnum/0.52)), int(round((risk2-0.48)*rnum/0.52))] = color
    #data = data[:,:,(rnum//2-1):,:]

    def update_line(num, data, ax, riskbarbottom, riskbartop):
        ax.imshow(data[:,:, num,:], origin='lower', extent=(0.0, 1.0, 0.0, 1.0))
        riskbarbottom.set_height(0.6*(num/(rnum-1)))
        riskbartop.set_height(0.6 - 0.6*(num/(rnum-1)))
        riskbartop.set_y(0.05 + 0.6*(num/(rnum-1)))
        return ax, riskbarbottom, riskbartop
    fig, ax = plt.subplots()
    ax.set_xlabel('Country 1 Hawkishness', size=14)
    ax.set_ylabel('Country 1 Risk Aversion', size=14)
    ax.text(-0.02,0,'Low', size=12, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.text(-0.02,1,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0,-0.02,'Low', size=12, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.text(1,-0.02,'High', size=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.tick_params(bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
    ax.text(1.16,0.7,'Risk Aversion 2', size=14, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    ax.text(1.18,0.65,'High', size=12, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.text(1.18,0.05,'Low', size=12, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    riskbarbottom = mpatches.Rectangle((1.15,0.05), 0.02, 0.3, color='limegreen', clip_on=False)
    riskbartop = ax.add_patch(mpatches.Rectangle((1.15,0.35), 0.02, 0.3, color='red', clip_on=False))
    ax.add_patch(riskbarbottom)
    ax.add_patch(riskbartop)
    ax.add_patch(mpatches.Rectangle((1.15,0.05), 0.02, 0.6, color='black', fill=False, clip_on=False))
    #plt.legend(handles=[mpatches.Patch(color='red', label='War'), mpatches.Patch(color='gold', label='Standoff'), mpatches.Patch(color='blue', label='Peace')], loc=(1.05, 0.8))
    plt.legend(handles=[mpatches.Patch(color='red', label='War'), mpatches.Patch(color='gold', label='Standoff'), mpatches.Patch(color='limegreen', label='Peace')], loc=(1.05, 0.8))
    #ax.imshow(data[:,:, 5,:], origin='lower', extent=(0.0, 1.0, 0.0, 1.0))
    line_ani = animation.FuncAnimation(fig, update_line, rnum, fargs=(data, ax, riskbarbottom, riskbartop),
                                      interval=10, blit=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.76, top=0.98)
    #plt.show()
    line_ani.save('figure_2.mp4', fps=10, bitrate=8192)

def twoplayeranimfile():
    X = np.loadtxt('data.csv', delimiter=',')
    print(np.shape(X))
    data1 = np.zeros((hnum+1,hnum+1,rnum+1, 3))
    #data2 = np.zeros((hnum+1,hnum+1,rnum+1, 3))
    for hawk1, hawk2, risk1, risk2, war, not_peace in X:
        color1 = (war, not_peace - war, 0)
        #color2 = colorsys.hsv_to_rgb((1.0/6.0) * (not_peace - war - war + 1.0) / 2.0, 1.0, not_peace)
        data1[round(hawk1*hnum), round(hawk2*hnum), round(risk1*rnum)] = color1
        #data2[round(hawk1*hnum), round(hawk2*hnum), round(risk1*rnum)] = color2

    def update_line(num, data, ax):
        ax.imshow(data[:,:, num,:], origin='lower', extent=(0.0, 1.0, 0.0, 1.0))
        return ax,
    fig, ax = plt.subplots()
    line_ani = animation.FuncAnimation(fig, update_line, rnum+1, fargs=(data1, ax),
                                       interval=50, blit=False)
    line_ani.save('rgb.mp4', fps=5)
    plt.close()

    #fig, ax = plt.subplots()
    #line_ani = animation.FuncAnimation(fig, update_line, rnum+1, fargs=(data2, ax),
    #                                  interval=50, blit=False)
    #line_ani.save('hsv.mp4', fps=5)
    #plt.close()
