import numpy as np
import matplotlib.pyplot as plt

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
    return inverse_utility(pWin * utility(2 + h, r) + (1-pWin) * utility(1 - h, r), r) - 1

def deal(pWin, hA, hB, rA, rB):
    wantA = expect(pWin, hA, rA)
    wantB = 1 - expect(1-pWin, hB, rB)
    print(wantA, wantB)
    return wantA <= wantB

def diagram():
    pWin = 0.5
    rA = 0.1
    rB = 0.1
    hA = np.array([[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]*10)
    hB = np.array([[0]*10, [0.1]*10, [0.2]*10, [0.3]*10, [0.4]*10, [0.5]*10, [0.6]*10, [0.7]*10, [0.8]*10, [0.9]*10])
    print(hA, hB)
    print(deal(pWin, hA, hB, rA, rB))
    
if __name__ == "__main__":
    diagram()
    