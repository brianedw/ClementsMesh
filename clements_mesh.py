# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Python Libraries
# %% [markdown]
# Standard Python imports

# %%
from UtilityMath import (plotComplexArray,
                         RandomComplexCircularMatrix, RandomComplexGaussianMatrix,
                         PolarPlot,
                         RescaleToUnitary,
                         ReIm,
                         MatrixSqError, MatrixError, MatrixErrorNormalized)
from scipy.optimize import minimize
from numpy import cos, sin, exp
import pylab as plt
import networkx as nx
from toolz.functoolz import (curry, pipe, thread_first)
from toolz.itertoolz import (concat, partition, interleave)
import scipy as sp
import numpy as np  # Does high performance dense array operations
import os
import sys
import time
import glob


# %%
from collections import deque
from pprint import pprint


# %%
from math import pi, radians
deg = radians(1)    # so that we can refer to 90*deg
I = 1j              # potentially neater imaginary nomenclature.


# %%
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))


# %%
# from toolz.dicttoolz import ()


# %%


# %%

# %% [markdown]
# # Work
# %% [markdown]
# ## Clements Decomposition

# %%


# %%
np.set_printoptions(edgeitems=30, linewidth=100000,
                    suppress=True,
                    formatter=dict(float=lambda x: "%.2g" % x))

# %% [markdown]
# ### Unitary Matrix Generation

# %%


def isPassive(M, verbose=False):
    Im = np.identity(M.shape[0])
    TH = np.conj(M.T)
    eigVals = np.linalg.eigvals(Im - TH @ M).real
    if verbose:
        print(eigVals)
    isPassive = np.alltrue(eigVals >= -1e-12)
    return isPassive


# %%
testM = np.identity(5, dtype='complex')
isPassive(testM)


# %%
def getRandomUnitaryMatrix(n=5, verbose=False):
    rMat = RandomComplexCircularMatrix(1, (n, n))
    # print(rMat)
    U, Svec, Vh = np.linalg.svd(rMat, full_matrices=True)
    S = np.diag(Svec)
    recovery = U @ S @ Vh
    if verbose:
        print("Successful SVD Decomposition:", np.allclose(recovery, rMat))
    M = U @ Vh
    return M


# %%
U = getRandomUnitaryMatrix(n=5)

# %% [markdown]
# ### Mixer
# %% [markdown]
# The `Mixer` represents a 4 port directional coupler with variable weights similar to a Mach-Zender Interferometer.
# %% [markdown]
# We begin with the definition of a directional "beam splitter" given below
# $$
# T(\theta, \phi) = \begin{pmatrix}
# e^{i\phi} \cos(\theta) & -\sin(\theta)\\
# e^{i\phi} \sin(\theta) & \cos(\theta)
# \end{pmatrix}
# $$
# This shows only the transmissive part and all reflections are assumed to be zero.

# %%


def T(theta, phi):
    a = [[exp(I*phi) * cos(theta), -sin(theta)],
         [exp(I*phi) * sin(theta),  cos(theta)]]
    return np.array(a)

# %% [markdown]
# We also need the inverse of this matrix.  This was computed in Mathematica.

# %%


def Tinv(theta, phi):
    a = [[exp(-I*phi) * cos(theta),  exp(-I*phi) * sin(theta)],
         [-sin(theta),                cos(theta)]]
    return np.array(a)

# %% [markdown]
# This allows us to define a simple class for a `Mixer`.

# %%


class Mixer:

    def __init__(self, theta_phi=(0, 0)):
        self.theta, self.phi = theta_phi

    def T(self):
        return T(self.theta, self.phi)

    def Tinv(self):
        return Tinv(self.theta, self.phi)


# %%
mixer = Mixer()
mixer = Mixer(theta_phi=(0.2*pi, 0.3*pi))


# %%
print(mixer.T())
print(mixer.Tinv())

# %% [markdown]
# ### Mixer Mesh Interaction
# %% [markdown]
# The formulation makes use of how a mixer interacts with all of the channels in the system.  Let us imagine a vector of $n$
#  lines.  Of course, if the mixer is not connected to a line, that line would remain unperturbed.  It follows that we can understand a mixer's effect on a
# mesh as the identity matrix with several elements changed to those given by $T(\theta, \phi)$ above.
#
# Note that within the paper, Clements uses the notation $T_{m,n}(\theta, \phi)$ which represents a mixing between lines $m$ and $n$.  In all cases within
# that work, $n = m + 1$.

# %%


def TMesh(theta_phi, N, lines):
    A = np.identity(N, dtype='complex')
    a = T(*theta_phi)
    m, n = lines
    A[m, m] = a[0, 0]
    A[m, n] = a[0, 1]
    A[n, m] = a[1, 0]
    A[n, n] = a[1, 1]
    return A


# %%
def TinvMesh(theta_phi, N, lines):
    A = np.identity(N, dtype='complex')
    a = Tinv(*theta_phi)
    m, n = lines
    A[m, m] = a[0, 0]
    A[m, n] = a[0, 1]
    A[n, m] = a[1, 0]
    A[n, n] = a[1, 1]
    return A


# %%
mixer = Mixer(theta_phi=(0.2*pi, 0.3*pi))
print(mixer.T())
print(mixer.Tinv())


# %%
TMesh(theta_phi=(0.3*pi, 0.2*pi), N=5, lines=(1, 2))


# %%
TinvMesh(theta_phi=(0.3*pi, 0.2*pi), N=5, lines=(1, 2))

# %% [markdown]
# And just as a check, let us verify that the analytical aligns with the numerical inverse.

# %%
invNum = np.linalg.inv(TMesh(theta_phi=(0.4*pi, 0.2*pi), N=5, lines=(1, 4)))
invAnal = TinvMesh(theta_phi=(0.4*pi, 0.2*pi), N=5, lines=(1, 4))
np.allclose(invNum, invAnal)

# %% [markdown]
# ### Mesh

# %%


def EvenQ(n):
    """
    True if n is even.
    False otherwise.
    """
    return(n % 2 == 0)


def OddQ(n):
    """
    True if n is odd.
    False otherwise.
    """
    return(n % 2 == 1)

# %% [markdown]
# Computes the number of layers in the mesh required to obtain `totCount` number of mixers.  For instance, suppose that you want 6 DoFs from a 4 port to 4 port mesh.  The even columns (0th, 2nd, etc) would have 2 mixers.  The odd columns (1st, 3rd, etc) would have 1 mixer.

# %%


def computeNLayers(evenCount, oddCount, totCount):
    """
    Given finds the total number of elements in the pattern [e, o, e, o, ...]
    required to achieve `totCount`.  Returns `None` if not evenly divisable.

    This is done directly by totaling the contribution of an e + o combination
    """
    if totCount == 1:
        return 1
    comboCount = evenCount + oddCount
    nComboLayers = totCount//comboCount
    if nComboLayers*comboCount == totCount:
        return 2*nComboLayers
    elif nComboLayers*comboCount + evenCount == totCount:
        return 2*nComboLayers + 1
    else:
        print("does not appear evenly divisable")
        return None


# %%
4 == computeNLayers(evenCount=2, oddCount=1, totCount=6)  # 2 + 1 + 2 + 1


# %%
3 == computeNLayers(evenCount=2, oddCount=5, totCount=9)  # 2 + 5 + 2


# %%
# None == computeNLayers(evenCount=2, oddCount=5, totCount=8) # 2 + 5 + ?

# %% [markdown]
# Generates Device Labels for mesh which can realize a given kernel size.
# It will return labels for each input port, output port, mixer, and thru.  Only the mixers
# are used for mesh calcu
#
# This is done in a naive fashion, stepping across the mesh from the input ports to
# the output ports.  For lines which do not intersect a mixer, a thru will be inserted.

# %%
def generateDeviceLabels(kernelSize, mixerLabel='m', thruLabel='t', inputLabel='i', outputLabel='o', verbose=False):
    NN = kernelSize
    evenCount = NN//2
    oddCount = (NN-1)//2
    if verbose:
        print("NN:", NN)
    nMixers = NN*(NN-1)//2
    if verbose:
        print("nDOFs:", nMixers)
    if verbose:
        print("evenCounts:", evenCount, "\toddCounts:", oddCount)
    nLayers = computeNLayers(evenCount, oddCount, nMixers)
    if verbose:
        print("nLayers:", nLayers)
    mixers = []
    thrus = []
    inPorts = [(inputLabel, i) for i in range(kernelSize)]
    if verbose:
        print("inPorts:", inPorts)
    outPorts = [(outputLabel, i) for i in range(kernelSize)]
    if verbose:
        print("outPorts:", outPorts)
    (i, j) = (0, 0)
    while i < nLayers:
        oddLayer = (i % 2 == 1)
        if (j == 0 and oddLayer) or (j == NN - 1):
            thrus.append((thruLabel, i, j))
            j += 1
        else:
            mixers.append((mixerLabel, i, j))
            j += 2
        if j >= NN:
            j = 0
            i += 1
    if verbose:
        print("mixers:", mixers)
    if verbose:
        print("thrus:", thrus)
    return (inPorts, mixers, thrus, outPorts)


# %%
(inPorts, mixers, thrus, outPorts) = generateDeviceLabels(
    kernelSize=5, mixerLabel='m', thruLabel='t', verbose=True)

# %% [markdown]
# Generates Mixer Labels for a mesh which can realize a given kernel size.
#
# This is done in a naive fashion, stepping across the mesh from the input ports
# to the output ports.  Mixers are grouped by column.  In other words, all mixers
# which touch the input ports are in the first list.

# %%


def generateMixerLabels(kernelSize, mixerLabel='m', verbose=False):
    NN = kernelSize
    evenCount = NN//2
    oddCount = (NN-1)//2
    if verbose:
        print("NN:", NN)
    nMixers = NN*(NN-1)//2
    if verbose:
        print("nDOFs:", nMixers)
    if verbose:
        print("evenCounts:", evenCount, "\toddCounts:", oddCount)
    nLayers = computeNLayers(evenCount, oddCount, nMixers)
    if verbose:
        print("nLayers:", nLayers)
    mixers = []
    (i, j) = (0, 0)
    for i in range(nLayers):
        if EvenQ(i):
            colList = [(mixerLabel, i, 2*j) for j in range(0, evenCount)]
        if OddQ(i):
            colList = [(mixerLabel, i, 2*j + 1) for j in range(0, oddCount)]
        mixers.append(colList)
    return (mixers)


# %%
generateMixerLabels(kernelSize=5, mixerLabel='m', verbose=True)

# %% [markdown]
# Generates Mixer Labels for a mesh which can realize a given kernel size.
#
# This is done on the diagonal to create a list which corresponds to the order in
# which they are nulled according to Clements et al.  Mixers are grouped by diagonal.

# %%


def generateMixerLabelsDiag(kernelSize, mixerLabel='m', verbose=False):
    NN = kernelSize
    orderedLabels = []
    for i in range(NN-1):
        diagList = []
        for j in range(i+1):
            if EvenQ(i):
                label = (mixerLabel, j, i - j)
            else:
                label = (mixerLabel, NN - j - 1, NN - (i - j) - 2)
            diagList.append(label)
        orderedLabels.append(diagList)
    return orderedLabels


# %%
generateMixerLabelsDiag(kernelSize=5, mixerLabel='m', verbose=False)

# %% [markdown]
# We now have three ways of generating mixer labels.  Let's confirm they all give the same results.

# %%
NN = 100
(inPorts, mixers, thrus, outPorts) = generateDeviceLabels(
    kernelSize=NN, mixerLabel='m', thruLabel='t', verbose=False)
s1 = mixers
s2 = list(concat(generateMixerLabels(
    kernelSize=NN, mixerLabel='m', verbose=False)))
s3 = list(concat(generateMixerLabelsDiag(
    kernelSize=NN, mixerLabel='m', verbose=False)))


# %%
sorted(s1) == sorted(s2) and sorted(s1) == sorted(s3)

# %% [markdown]
# ### Drawing

# %%
(inPortLabels, mixerLabels, thruLabels, outPortLabels) = generateDeviceLabels(
    kernelSize=5, mixerLabel='m', thruLabel='t', verbose=True)


# %%
def makeGraph(inPortLabels, mixerLabels, thruLabels, outPortLabels):
    G = nx.Graph()
    maxMixerX = np.max(np.array(mixerLabels, dtype=object)[:, 1])
    maxMixerY = np.max(np.array(mixerLabels, dtype=object)[:, 2])
    for lab in inPortLabels:
        (_, y) = lab
        G.add_node(lab, pos=(-1, -y), col='#88ff88')
    for lab in mixerLabels:
        (_, x, y) = lab
        G.add_node(lab, pos=(x, -y-0.5), col='#ffff00')
    for lab in thruLabels:
        (_, x, y) = lab
        if y == 0:
            G.add_node(lab, pos=(x, -y-0.5), col='#8888ff')
        else:
            G.add_node(lab, pos=(x, -y+0.5), col='#8888ff')
    for lab in outPortLabels:
        (_, y) = lab
        G.add_node(lab, pos=(maxMixerX+1, -y), col='#ff8888')
    for lab in inPortLabels:
        pass
    allElements = set()
    for l in (inPortLabels, mixerLabels, thruLabels, outPortLabels):
        allElements.update(l)
    for lab in mixerLabels:
        (_, x, y) = lab
        potLab = ('m', x+1, y+1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('m', x+1, y-1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', x+1, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', x+1, y+1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', x-1, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', x-1, y+1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
    for lab in inPortLabels:
        (_, y) = lab
        potLab = ('m', 0, y-1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('m', 0, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', 0, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
    for lab in outPortLabels:
        (_, y) = lab
        potLab = ('m', maxMixerX, y-1)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('m', maxMixerX, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
        potLab = ('t', maxMixerX, y)
        if potLab in allElements:
            G.add_edge(lab, potLab)
    return G


# %%
def plotMesh(kSize):
    (inPortLabels, mixerLabels, thruLabels, outPortLabels) = generateDeviceLabels(
        kernelSize=kSize, mixerLabel='m', thruLabel='t', verbose=False)
    G = makeGraph(inPortLabels, mixerLabels, thruLabels, outPortLabels)
    pos = nx.get_node_attributes(G, 'pos')
    colors = nx.get_node_attributes(G, 'col')
    nx.draw(G, pos, with_labels=True, node_size=2000,
            font_size=10, node_color=list(colors.values()))
    fig = plt.figure(1, figsize=(20, 10), dpi=60)
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()


# %%
plotMesh(5)

# %% [markdown]
# ### Solving
# %% [markdown]
# Now let's apply this to a random unitary matrix of size `NN`.

# %%
NN = 5


# %%
U = getRandomUnitaryMatrix(n=NN)


# %%
def ChopPrint(A, tol=1e-16):
    AC = A.copy()
    AC.real[abs(AC.real) < tol] = 0.0
    AC.imag[abs(AC.imag) < tol] = 0.0
    print(AC)

# %% [markdown]
# As part the Clements iterative algorithm, a matrix product is nulled element by element by
# tweaking the $\theta$ and $\phi$ on a mixer while the effect of that mixer is applied to either
# the left or right side of the matrix product.  For each iteration, we need to know:
# * which mixer is being tweaked
# * whether it is being applied to the left or right side
# * which element of the matrix product is being nulled
# %% [markdown]
# The ordering of the mixers has already been computed and an example is shown below.

# %%


def buildMeshEquation(kSize, verbose=False):
    mixerLabelsDiag = generateMixerLabelsDiag(
        kernelSize=kSize, mixerLabel='m', verbose=False)
    frontLabelsM = mixerLabelsDiag[0::2]
    if verbose:
        print("frontLabelsM:", frontLabelsM)
    backLabelsM = mixerLabelsDiag[1::2]
    if verbose:
        print("backLabelsM:", backLabelsM)

    LHS = deque([{'item': 'D'}])
    RHS = deque()
    RHS.append({'item': 'U'})
    UPos = 0
    for listM in frontLabelsM:
        for label in listM:
            operatorDesc = {'item': label, 'inv': True}
            RHS.append(operatorDesc)
    for listM in backLabelsM:
        for label in listM:
            operatorDesc = {'item': label, 'inv': False}
            RHS.appendleft(operatorDesc)
            UPos += 1
    if verbose:
        print("LHS:")
    if verbose:
        pprint(LHS)
    if verbose:
        print("RHS:")
    if verbose:
        pprint(RHS)
    if verbose:
        print("UPos:", UPos)
    return (LHS, RHS)


# %%
(LHS, RHS) = buildMeshEquation(NN, verbose=True)


# %%
def generateMatrixZeroTargets(kernelSize):
    NN = kernelSize
    orderedLabels = []
    cMax = 0
    for i in range(1, NN):
        diagList = []
        cMin = 0
        cMax = i
        rMin = NN - i
        rMax = NN
        for r, c in zip(range(rMin, rMax), range(cMin, cMax)):
            label = (r, c)
            diagList.append(label)
        if OddQ(i):
            diagList.reverse()
        orderedLabels.append(diagList)
    return orderedLabels


# %%
generateMatrixZeroTargets(NN)


# %%
def buildSolultionRanges(UPos, kSize):
    leftRange = 0
    rightRange = 0

    solRange = []
    for n in range(1, kSize):
        # zeroTargL = zeroTargs[(n-1)]
        for i in range(1, n+1):
            # zeroTarg = zeroTargL[i-1]
            if OddQ(n):
                newItem = {'targMat': rightRange + 1 + UPos,
                           'matRange': (leftRange + UPos, rightRange + UPos + 1),
                           'side': 'R'}
                solRange.append(newItem)
                rightRange += 1
            if EvenQ(n):
                newItem = {'targMat': leftRange - 1 + UPos,
                           'matRange': (leftRange + UPos, rightRange + UPos + 1),
                           'side': 'L'}
                solRange.append(newItem)
                leftRange -= 1
    return solRange


# %%
UPos = RHS.index({'item': 'U'})
buildSolultionRanges(UPos, NN)


# %%
def buildSolvingPath(kSize, UPos):
    solRanges = buildSolultionRanges(UPos, kSize)
    zeroTargs = concat(generateMatrixZeroTargets(kernelSize=kSize))
    mixerLabels = concat(generateMixerLabelsDiag(
        kernelSize=kSize, mixerLabel='m', verbose=False))

    solvingPath = []
    for solRange, zeroTarg, mixerLabel in zip(solRanges, zeroTargs, mixerLabels):
        step = {}
        step['targMat'] = solRange['targMat']
        step['matRange'] = solRange['matRange']
        step['side'] = solRange['side']
        step['targElem'] = zeroTarg
        step['mixer'] = mixerLabel
        solvingPath.append(step)
    return solvingPath


# %%
solvingPath = buildSolvingPath(NN, UPos)
solvingPath

# %% [markdown]
# We can create a dictionary of mixers using the labels as keys.

# %%
mixerLabelsDiag = generateMixerLabelsDiag(NN)
mixerDict = {label: Mixer() for label in concat(mixerLabelsDiag)}


# %%
D = np.identity(n=NN, dtype='complex')

# %% [markdown]
# For each item description, we can obtain a matrix.  As such, the LHS and RHS above
# represent matrix products.

# %%


def getM(desc):
    if desc['item'] == 'U':
        M = U
    elif desc['item'] == 'D':
        M = D
    elif desc['item'][0] == 'm':
        (_, i, j) = desc['item']
        mixer = mixerDict[desc['item']]
        theta_phi = (mixer.theta, mixer.phi)
        if(desc['inv'] == False):
            M = TMesh(theta_phi, N=NN, lines=(j, j+1))
        else:
            M = TinvMesh(theta_phi, N=NN, lines=(j, j+1))
    else:
        print("You shouldn't be here")
    return M


# %%
# for desc in LHS:
#     print(getM(desc, mixerDict, kSize, U, D))

# for desc in RHS:
#     print(getM(desc, mixerDict, kSize, U, D))


# %%
def partialProduct(RHS, lR, rR, getM, verbose=False):
    matrixList = []
    for i, term in enumerate(RHS):
        if i >= lR and i < rR:
            M = getM(term)
            matrixList.append(M)
    if verbose:
        pprint(matrixList)
    if len(matrixList) == 1:
        return matrixList[0]
    else:
        return np.linalg.multi_dot(matrixList)


# %%
partialProduct(RHS, 6, 8, getM, verbose=True)

# %% [markdown]
# We will use the `minimize` function above to find the proper $\theta$ and $\phi$
# for each mixer in turn.  Here `f` is a cost function which will be minimized by
# tweaking $\theta$ and $\phi$ while trying to render the element `zeroTarg` of
# $(S_{\mathrm{tot}} \times T^{-1})$ to be zero, assuming `side = 'r'`.  Note that if
# `side = 'l'`, then we will be looking at $(T \times S_{\mathrm{tot}})$

# %%


# %%
def f(theta_phi, Stot, side, zeroTarg, lines):
    NN, _ = Stot.shape
    if side == 'R':
        Ainv = TinvMesh(theta_phi, N=NN, lines=lines)
        Ttot = Stot@Ainv
    elif side == 'L':
        A = TMesh(theta_phi, N=NN, lines=lines)
        Ttot = A@Stot
    else:
        print("you shouldn't be here")
    return np.abs(Ttot[zeroTarg])


# %%
pprint(solvingPath)


# %%
def nullAllMixers(RHS, solvingPath, mixerDict, getM, verbose=False):
    for pathStep in solvingPath:
        if verbose:
            pprint(pathStep)

        lR, rR = pathStep['matRange']
        side = pathStep['side']
        zeroTarg = pathStep['targElem']
        mixerLabel = pathStep['mixer']
        (_, _, topLine) = mixerLabel
        lines = (topLine, topLine + 1)

        partProduct = partialProduct(RHS, lR, rR, getM)
        sol = minimize(f, x0=[0, 0], args=(partProduct, side, zeroTarg, lines))
        setThetaPhi = sol.x
        if verbose:
            print(mixerLabel, setThetaPhi, sol.fun)
        (_, i, j) = mixerLabel
        setMixer = mixerDict[mixerLabel]
        setMixer.theta, setMixer.phi = setThetaPhi


# %%
nullAllMixers(RHS, solvingPath, mixerDict, getM, verbose=True)

# %% [markdown]
# At this point, the RHS should be a diagonal matrix corresponding to $\mathbf{D}$

# %%
partialProduct(RHS, 0, len(RHS), getM)

# %% [markdown]
# There will be some minor numerical errors for off-diagonal terms which we will remove using element-by-element multiplication
# with an identity matrix

# %%
D[:] = np.identity(n=NN) * partialProduct(RHS, 0, len(RHS), getM)
print(D)

# %% [markdown]
# Next, we pop the matrices off the RHS to the LHS.  In typical fashion, this involves the inverse.  We begin with the equation below

# %%
pprint(LHS)
print(" == ")
pprint(RHS)

# %% [markdown]
# And then first strip the everything to the "right" of the $U$, and then everything to the "left" of the $U$.

# %%


def solveForU(LHS, RHS):
    while(True):
        term = RHS[-1]
        if term['item'] == 'U':
            break
        RHS.pop()
        term['inv'] = not term['inv']
        LHS.append(term)

    while(True):
        term = RHS[0]
        if term['item'] == 'U':
            break
        RHS.popleft()
        term['inv'] = not term['inv']
        LHS.appendleft(term)

    return (RHS, LHS)


# %%
LHS, RHS = solveForU(LHS, RHS)


# %%
pprint(LHS)
print(" == ")
pprint(RHS)

# %% [markdown]
# At this point, we have a system of mixers and an array of phase shifters nestled in the middle.  The product of all of these elements is indeed $\mathbf{U}$.

# %%
print("RHS:")
print(np.linalg.multi_dot(list(map(getM, RHS))))
print("U: ")
print(U)

# %% [markdown]
# However, we notice that approximately the last half of these elements (left-most) are defined by their inverse.  Additionally, the
# middle might not be a convenient place to put an array of phase shifters.  Next, we employ a transformation to push
# $\mathbf{D}$ to the very end of the structure, which in matrix multiplication would be the left-most term, while transforming
# the "inverse mixers" into standard ones.
# %% [markdown]
# The idea is that given a mixer with transmission coefficient:
#
# $$
# T(\theta, \phi) = \begin{pmatrix}
# e^{i\phi} \cos(\theta) & -\sin(\theta)\\
# e^{i\phi} \sin(\theta) & \cos(\theta)
# \end{pmatrix}
# $$
#
# and a phase shifter array
#
# $$
# D(\gamma_1, \gamma_2) = \begin{pmatrix}
# e^{i\gamma_1}  & 0\\
# 0 & e^{i\gamma_2}
# \end{pmatrix}
# $$
#
# and a sequence $(T(\theta, \phi))^{-1}\times D(\gamma_1, \gamma_2)$, can we
# find an equivalent $ D(\gamma_1', \gamma_2') \times T(\theta', \phi')$ and if so,
# is there a convenient transform from $\{\theta, \phi, \gamma_1, \gamma2 \}$ to
# $\{\theta', \phi', \gamma_1', \gamma2'\}$?
# %% [markdown]
# After a bit of work, one can conclude that, indeed, such a transformation can
# occur providing:
# \begin{align*}
# \theta' &= -\theta \\
# \phi' &= \gamma_1 - \gamma_2 \\
# \gamma_1' &= \gamma_2 - \phi \\
# \gamma_2' &= \gamma_2 \\
#
# \end{align*}
# %% [markdown]
# First we must find the location of the D matrix in the product:

# %%
RHS
DPos = next(i for (i, x) in enumerate(RHS) if x['item'] == 'D')
print(DPos)

# %% [markdown]
# Next we iteratively move D to the left, applying the transformtion rules above.

# %%


def pushOutD(RHS, mixerDict, D):
    DPos = next(i for (i, x) in enumerate(RHS) if x['item'] == 'D')

    while DPos > 0:
        # get mixer to left of D
        termDesc = RHS[DPos - 1]
        label = termDesc['item']
        (_, _, j) = label
        L1, L2 = (j, j+1)
        mixer = mixerDict[label]

        # get old mixer and D values for the connected lines.
        gamma1 = np.real(np.log(D[L1, L1])/I)
        gamma2 = np.real(np.log(D[L2, L2])/I)
        theta = mixer.theta
        phi = mixer.phi

        # computer new mixer and D values for the connected lines.
        thetap = -theta
        phip = gamma1 - gamma2
        gamma1p = gamma2 - phi
        gamma2p = gamma2

        # set the mixer to the new values.
        mixer.theta = thetap
        mixer.phi = phip
        termDesc['inv'] = False
        # set D to the new values.
        D[L1, L1] = exp(I*gamma1p)
        D[L2, L2] = exp(I*gamma2p)

        # swap the location of the mixer and D.
        (RHS[DPos], RHS[DPos-1]) = (RHS[DPos-1], RHS[DPos])
        DPos = DPos - 1


# %%
pushOutD(RHS, mixerDict, D)

# %% [markdown]
# And indeed, we see that D is in the front position.

# %%
RHS

# %% [markdown]
# and that the product of the above devices, does yield the original $\mathbf{U}$.

# %%
print("RHS:")
print(np.linalg.multi_dot(list(map(getM, RHS))))
print("U: ")
print(U)

# %% [markdown]
# ## Examination as Series of Banded Matrix Operators
# %% [markdown]
# Now let us reorder the elements according to their column position.

# %%
mixerLabelsRect = generateMixerLabels(5, mixerLabel='m', verbose=False)


# %%
termsRect = [{'item': 'D'}] + [{'item': label, 'inv': False}
                               for col in mixerLabelsRect[::-1] for label in col]
termsRect

# %% [markdown]
# And then verify that this does not effect the operation at all.

# %%
print(np.linalg.multi_dot(list(map(getM, termsRect[:]))))
print(U)

# %% [markdown]
# However, now we can determine the effect of applying only the first column.

# %%
termsRect[-2:]


# %%
print(np.linalg.multi_dot(list(map(getM, termsRect[-2:]))))

# %% [markdown]
# Or the first two columns

# %%
termsRect[-4:]


# %%
print(np.linalg.multi_dot(list(map(getM, termsRect[-4:]))))

# %% [markdown]
# # Defining a Class

# %%


class ClementsMesh:
    pass


# %%
def __init__(self, U, verbose=False):
    self.U = U
    if not isPassive(U):
        print("U does not appear to be passive")
        return
    kSize = U.shape[0]
    self.kSize = kSize

    mixerLabelsDiag = generateMixerLabelsDiag(kSize)
    self.mixerDict = {label: Mixer() for label in concat(mixerLabelsDiag)}
    self.D = np.identity(n=kSize, dtype='complex')

    (self.LHS, self.RHS) = buildMeshEquation(self.kSize, verbose=verbose)
    UPos = self.RHS.index({'item': 'U'})
    self.solvingPath = buildSolvingPath(kSize=self.kSize, UPos=UPos)

    # (inPortLabels, mixerLabels, thruLabels, outPortLabels) = generateDeviceLabels(kernelSize=self.kSize, mixerLabel='m', thruLabel='t', verbose=True)


setattr(ClementsMesh, "__init__", __init__)


# %%
def getM(self, desc):
    if desc['item'] == 'U':
        M = self.U
    elif desc['item'] == 'D':
        M = self.D
    elif desc['item'][0] == 'm':
        (_, i, j) = desc['item']
        mixer = self.mixerDict[desc['item']]
        theta_phi = (mixer.theta, mixer.phi)
        if(desc['inv'] == False):
            M = TMesh(theta_phi, N=self.kSize, lines=(j, j+1))
        else:
            M = TinvMesh(theta_phi, N=self.kSize, lines=(j, j+1))
    else:
        print("You shouldn't be here")
    return M


setattr(ClementsMesh, "getM", getM)


# %%
def printMeshEq(self):
    pprint(self.LHS)
    print(" == ")
    pprint(self.RHS)


setattr(ClementsMesh, "printMeshEq", printMeshEq)


# %%
def plotMeshC(self):
    plotMesh(self.kSize)


setattr(ClementsMesh, "plotMesh", plotMeshC)


# %%
def nullAllMixersC(self):
    nullAllMixers(self.RHS, self.solvingPath, self.mixerDict, self.getM)


setattr(ClementsMesh, "nullAllMixers", nullAllMixersC)


# %%
def solveForD(self, verbose=False):
    if verbose:
        self.printMeshEq()
    DMessy = partialProduct(self.RHS, 0, len(self.RHS), self.getM)
    if verbose:
        print(DMessy)
    DClean = np.identity(n=self.kSize) * DMessy
    self.D[:] = DClean
    if verbose:
        print(D)
    if verbose:
        self.printMeshEq()


setattr(ClementsMesh, "solveForD", solveForD)


# %%
def solveForUC(self, verbose=False):
    self.LHS, self.RHS = solveForU(self.LHS, self.RHS)


setattr(ClementsMesh, "solveForU", solveForUC)


# %%
def pushOutDC(self, verbose=False):
    pushOutD(self.RHS, self.mixerDict, self.D)
    if verbose:
        URealized = partialProduct(self.RHS, 0, len(self.RHS), self.getM)
        workedQ = np.allclose(URealized, self.U)
        if workedQ:
            print("worked!")
        else:
            print("didn't work")
            pprint(URealized - self.U)


setattr(ClementsMesh, "pushOutD", pushOutDC)


# %%
def calibrate(self):
    self.nullAllMixers()
    self.solveForD()
    self.solveForU()
    self.pushOutD()


setattr(ClementsMesh, "calibrate", calibrate)


# %%
def getLayerTransferFunction(self, lS, lE):
    mixerLabelsRect = generateMixerLabels(
        self.kSize, mixerLabel='m', verbose=False)
    mixerLabels
    termsRect = [{'item': 'D'}] + [{'item': label, 'inv': False}
                                   for col in mixerLabelsRect[::-1] for label in col]


# %%
def getLayerTransferFunction(self, layStart, layEnd, verbose=False):
    mixerLabelsRect = [
        'D'] + generateMixerLabels(self.kSize, mixerLabel='m', verbose=False)[::-1]
    if verbose:
        pprint(mixerLabelsRect)
    selLabels = list(concat(mixerLabelsRect[-layEnd: -layStart or None]))
    terms = []
    for label in selLabels:
        if label == 'D':
            terms.append({'item': 'D'})
        else:
            terms.append({'item': label, 'inv': False})
    if verbose:
        pprint(terms)
    T = partialProduct(terms, 0, len(terms), self.getM)
    return T


setattr(ClementsMesh, "getLayerTransferFunction", getLayerTransferFunction)

# %% [markdown]
# ## Verbose Class Example

# %%
U = getRandomUnitaryMatrix(n=6)


# %%
mesh = ClementsMesh(U)


# %%
mesh.printMeshEq()


# %%
mesh.solvingPath


# %%
mesh.nullAllMixers()


# %%
mesh.plotMesh()


# %%
mesh.solveForD(verbose=True)


# %%
mesh.solveForU()


# %%
mesh.pushOutD(verbose=True)


# %%
mesh.printMeshEq()


# %%
mesh.getLayerTransferFunction(0, 7)


# %%
U

# %% [markdown]
# ## Compact Class Example

# %%
U = getRandomUnitaryMatrix(n=10)


# %%
mesh = ClementsMesh(U)
mesh.calibrate()


# %%
mesh.plotMesh()


# %%
mesh.getLayerTransferFunction(0, 11)


# %%
U


# %%
