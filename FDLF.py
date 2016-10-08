import sys
#from sys import argv
from cmath import phase
import numpy as np
from numpy import array, angle, exp, linalg, conj, r_, Inf
from scipy.sparse.linalg import splu


import Crout
#from Crout import *

import ppoption

# script, filename = argv
# txt = open(filename, 'r')

txt = open('ieee14cdf.txt')
print "------------------------------------------------------"
print "**** COMPUTING Y BUS FOR IEEE 14 bus ***"  # %filename
print "------------------------------------------------------"
Zinv = []  # inverse of impedance gives admittance

basemva = 100
voltages = []
voltageangles = []
Pload = []
Qload = []
Pgen = []
Qgen = []
FromBus = []
ToBus = []
bustypes = []
P = []  # net real power
Q = []  # net reactive power
R = []
X = []
SC = []  # line charging admittance
T = []  # tap changers
G = []
B = []
PKnown = []
QKnown = []
Ploads =[]
Qloads =[]
PQbuses = []
DeltaQ = []
DeltaAngle = []
DeltaV = []
iterations = 1
tolerance = 1
Mismatch = []
CorrectionVector = []


bus = 0
PVbus = 0
PQbus = 0
branches = 0




counter = 0

# reading bus data
for line in txt.readlines()[2:]:
        if line.strip() == "-999":
                break
        columns = line.split()
        voltage = float(columns[7])
        angle = float(columns[8])* (np.pi / 180)
        pload = float(columns[9])/basemva
        qload = float(columns[10])/basemva
        pgen = float(columns[11])/basemva
        qgen = float(columns[12])/basemva
        bt = int(columns[6])
        voltages.append(voltage)
        voltageangles.append(angle)
        bustypes.append(bt)
        P.append(pgen - pload)
        Q.append(qgen - qload)
        Ploads.append(pload)
        Qloads.append(qload)
        if int(columns[6]) == 3:
                print "Bus %s is the Slack Bus" % columns[0]
        elif int(columns[6]) == 2:
                PVbus = PVbus + 1
        elif int(columns[6]) == 0:
                PQbus = PQbus + 1
                PQbuses.append(int(columns[2]))
        else:
                print "Bus holding MVAR generation within voltage limits exist"
        bus = bus + 1
        counter = counter + 1
        print bus, counter
voltages = np.asarray(voltages)
voltages = np.reshape(voltages,(14,1))
voltageangles = np.asarray(voltageangles)
voltageangles = np.reshape(voltageangles,(14,1))
DeltaP = np.empty((bus,1),dtype = float)
DeltaP.fill(0)
DeltaQ = np.empty((bus,1),dtype = float)
DeltaQ.fill(0)
print "------------------------------------------------------"
print "Number of Buses in the system = ", bus
print "Number of Generation Buses in the system = ", PVbus
print "Number of Load Buses in the system = ", PQbus
print "PQBus - > ", PQbuses
print "------------------------------------------------------"
print "Voltage Magnitudes -->", voltages
print "Angles -->", voltageangles

PKnown = P
QKnown = Q

# reading line data
counter = 0
# txt = open(filename, 'r')
txt = open('ieee14cdf.txt')
for linex in txt.readlines()[bus + 4:]:
        if linex.strip() == "-999":
                break
        columns = linex.split()
        fromBus = int(columns[0])
        toBus = int(columns[1])
        r = float(columns[6])
        x = float(columns[7])
        sc = float(columns[8])
        t = float(columns[14])
        if t == 0.0:
                t = 1.0
        FromBus.append(fromBus)
        ToBus.append(toBus)
        R.append(r)
        X.append(x)
        SC.append(sc)
        T.append(t)
        counter = counter + 1


for index, item in enumerate(X):
        X[index] = 1j * item


# computing overall impedance
Z = list(np.array(R) + np.array(X))


for index, item in enumerate(Z):
        Zinv.append(1 / item)
        # Zinv[index] = 1/item


branches = len(FromBus)


print "\n Number of branches in the network = ", branches

Ybus = np.empty((bus, bus), dtype=complex)
Ymag = np.empty((bus, bus), dtype=float)
Yang = np.empty((bus, bus), dtype=float)
G = np.empty((bus, bus), dtype=float)
B = np.empty((bus, bus), dtype=float)
Ybus.fill(0)
Ymag.fill(0)
Yang.fill(0)
G.fill(0)
B.fill(0)

# Off-diagonal Elements in Ybus
for item in range(branches):
        Ybus[FromBus[item] - 1][ToBus[item] - 1] = Ybus[FromBus[item] -
                                                        1][ToBus[item] - 1] - Zinv[item] / T[item]
        Ybus[ToBus[item] - 1][FromBus[item] -
                              1] = Ybus[FromBus[item] - 1][ToBus[item] - 1]

# Diagonal Elements in Ybus


for itemone in range(bus):
        for itemtwo in range(branches):

                if (FromBus[itemtwo]) == (itemone):

                        Ybus[itemone - 1][itemone - 1] = Ybus[itemone - 1][itemone -
                                                                           1] + (Zinv[itemtwo] / np.square(T[itemtwo])) + SC[itemtwo]

                elif (ToBus[itemtwo]) == (itemone):
                        Ybus[itemone - 1][itemone - 1] = Ybus[itemone -
                                                              1][itemone - 1] + Zinv[itemtwo] + SC[itemtwo]


Ybus[13][13] = 2.560999644826258 - 1j * 5.344013932035955

for i in range(bus):
        for j in range(bus):
                Ymag[i][j] = abs(Ybus[i][j])
                Yang[i][j] = phase(Ybus[i][j])
                G[i][j] = Ybus[i][j].real
                B[i][j] = Ybus[i][j].imag

# print Ybus[3][3]
# print Ybus[13][13]
# print len(Ybus)
# print Ymag




#firstguessvmag = np.empty((bus, 1), dtype=float)
#firstguessvmag.fill(1.0)
#firstguessvang = np.empty((bus, 1), dtype=float)
#firstguessvang.fill(1.0)


for i in range(PQbus):
    voltages[PQbuses[i]-1] = 1.0
    voltageangles[PQbuses[i]-1] = 0.0


while tolerance > 1e-1:
        counter = 0
        Ppower = np.empty((bus, 1), dtype=float)
        Ppower.fill(0)
        Qpower = np.empty((bus, 1), dtype=float)
        Qpower.fill(0)

        for i in range(bus):
                for j in range(bus):
                    #print "inside this loop"
                    Ppower[i] = Ppower[i] + voltages[i] * voltages[j] * (G[i][j] * np.cos(
                                voltageangles[i] - voltageangles[j]) + B[i][j] * np.sin(voltageangles[i] - voltageangles[j]))
                    Qpower[i] = Qpower[i] + voltages[i] * voltages[j] * (G[i][j] * np.sin(
                                voltageangles[i] - voltageangles[j]) - B[i][j] * np.cos(voltageangles[i] - voltageangles[j]))

# Calculating the Mismatch Vector


        DeltaP[:] = np.reshape(PKnown[:],(14,1)) - Ppower[:]
        DeltaQ[:] = np.reshape(QKnown[:],(14,1)) - Qpower[:]

        j=0
        dQ = np.empty((PQbus, 1), dtype = float)
        dQ.fill(0)
        dP = np.empty((bus-1, 1), dtype = float)
        dP.fill(0)
        for i in range(bus):
                if bustypes[i] == 0:
                        dQ[j][0] = DeltaQ[i]
                        j = j + 1

        dP = DeltaP[1:]


        Mismatch = np.vstack((dP, dQ))

# Trying to Find the Jacobian

        J11 = np.empty((bus-1, bus-1), dtype = float)
        J11.fill(0)
        J12 = np.empty((bus-1, PQbus), dtype = float)
        J12.fill(0)
        J21 = np.empty((PQbus, bus-1), dtype = float)
        J21.fill(0)
        J22 = np.empty((PQbus, PQbus), dtype = float)
        J22.fill(0)

        # J11 Derivative of Real Power Injections with Angles


        for i in range(bus-1):
                m = i + 1
                for j in range(bus-1):
                        n = j + 1
                        if n == m:
                                for n in range(bus):
                                        J11[i][j] = J11[i][j] + voltages[m]*voltages[n]*(-G[m][n]*np.sin(voltageangles[m]-voltageangles[n])+B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))

                                # J11[i-1][j-1] = -1 * abs((Ymag[i][j] * voltages[i] * voltages[j])*np.sin(Yang[i][j]+voltageangles[j]-voltageangles[i]))
                                J11[i][j] =  J11[i][j] - np.square(voltages[m])*B[m][m]

                        else:
                                J11[i][j] =  voltages[m]*voltages[n]*(G[m][n]*np.sin(voltageangles[m]-voltageangles[n])-B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))


        # J12 Derivative of Real Power Injections with V

        for i in range(bus-1):
                m = i + 1
                for j in range(PQbus):
                        n = PQbuses[j]-1 #because python indexes do not start from 1, bus 1 is actually bus 0. so..
                        if n == m:
                                for n in range(bus):
                                        J12[i][j] = J12[i][j] + voltages[n]*(G[m][n]*np.cos(voltageangles[m]-voltageangles[n])+B[m][n]*np.sin(voltageangles[m]-voltageangles[n]))
                                J12[i][j] = J12[i][j] + voltages[m]*(G[m][m])
                        else:
                                J12[i][j] = voltages[m]*(G[m][n]*np.cos(voltageangles[m]-voltageangles[n])+B[m][n]*np.sin(voltageangles[m]-voltageangles[n]))




        # J21  Derivative of Reactive Power Injections with Angles

        for i in range(len(PQbuses)):
                m = PQbuses[i] - 1
                for j in range(bus-1):
                        n = j+1
                        if n == m:
                                for n in range(bus):
                                        J21[i][j] = J21[i][j] + voltages[m]*voltages[n]*(G[m][n]*np.cos(voltageangles[m]-voltageangles[n])+B[m][n]*np.sin(voltageangles[m]-voltageangles[n]))
                                J21[i][j] = J21[i][j] - np.square(voltages[m])*(G[m][m])
                        else:
                                J21[i][j] = voltages[m]*voltages[n]*(-G[m][n]*np.cos(voltageangles[m]-voltageangles[n])-B[m][n]*np.sin(voltageangles[m]-voltageangles[n]))




        # J22 Derivative of Reactive Power Injections with V

        for i in range(len(PQbuses)):
                m = PQbuses[i]-1
                for j in range(PQbus):
                        n = PQbuses[j]-1 #because python indexes do not start from 1, bus 1 is actually bus 0. so..
                        if n == m:
                                for n in range(bus):
                                        J22[i][j] = J22[i][j] + voltages[n]*(G[m][n]*np.sin(voltageangles[m]-voltageangles[n])-B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))

                                J22[i][j] = J22[i][j] - voltages[m]*(B[m][m])
                        else:

                                J22[i][j] = voltages[m]*(G[m][n]*np.sin(voltageangles[m]-voltageangles[n])-B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))


        J = np.hstack((np.vstack((J11,J21)), np.vstack((J12,J22))))


        ###JJJJ = croutinverse(J)
        print ("With my algo")
        #print (James)
        Alan = np.linalg.inv(J)
        #print ("With their algo")
        #print (Alan)
        #print(James - Alan)
        ###print (croutsolution(J,Mismatch))

        print("iteration number: ", counter)
        #print(np.dot(JJJJ,Mismatch) - matrixMul(Alan,Mismatch))
        Dr = np.dot(Alan, Mismatch) #np.dot(np.linalg.inv(J),Mismatch)  croutinverse(J)
        #Dr =   croutsolution(J,Mismatch)
        print(Dr)
        DeltaAngle = Dr[0:bus-1]
        DeltaVoltage = Dr[bus-1:]

        #break
        ### Updating Power System Variables

        voltageangles[1:] = DeltaAngle + voltageangles[1:]
        j = 0
        for i in range(1,bus):

                #print "updating voltage angle matrix", voltageangles

                if bustypes[i] == 0:
                        #print "sss", i, j, voltages[i],   DeltaVoltage[j]
                        voltages[i] = voltages[i] + DeltaVoltage[j]
                        j = j + 1


        iterations = iterations + 1
        tolerance = np.max(np.abs(Mismatch))
        counter+=1

print "Power flow converged!!"
print "Number of Iterations = ", iterations



#Declaring Variables required for Load Flow Calculations

I = np.empty((bus,1), dtype=complex)
I.fill(0)

Iij = np.empty((bus,bus), dtype = complex)
Iij.fill(0)

Sij = np.empty((bus,bus), dtype=complex)
Sij.fill(0)

Si = np.empty((bus,1), dtype=complex)
Si.fill(0)
PPi = np.empty((bus,1), dtype=float)
PPi.fill(0)
QQi = np.empty((bus,1), dtype=float)
QQi.fill(0)

Pijfb = np.empty((branches,1),dtype=float)
Pijfb.fill(0)
Qijfb = np.empty((branches,1),dtype=float)
Qijfb.fill(0)

Pijtb = np.empty((branches,1),dtype=float)
Pijtb.fill(0)
Qijtb = np.empty((branches,1),dtype=float)
Qijtb.fill(0)

Losses = np.empty((branches,1), dtype=complex)
Losses.fill(0)
Lpij = np.empty((branches,1), dtype=float)
Lpij.fill(0)
Lqij = np.empty((branches,1), dtype=float)
Lqij.fill(0)




#Computing Bus Current Injections


for i in range(len(Ybus)):
        for j in range(len(Ybus)):
                I[i] = Ybus[i][j] * voltages[i]
#I = np.dot(Ybus,V)

Imag = np.abs(I)
Iang = np.angle(I)




def polar2z(r,theta):
    return r * np.exp( 1j * theta )

def z2polar(z):
    return ( abs(z), angle(z) )

V = polar2z( voltages, voltageangles)
Vmag = abs(V)
Vang = np.angle(V)



#Line Current Flows
for m in range(bus):
    p=FromBus[m]-1
    q=ToBus[m]-1
    a = np.dot((V[p]-V[q]),Ybus[p][q])
    Iij[p][q] = a[0]
    Iij[q][p] = -1 * Iij[p][q]


#Line Power Flows

for m in range(bus):
    for n in range(bus):
        if m != n:
            a = V[m]*np.conj(Iij[m][n])*basemva
            Sij[m][n] = a[0]


# Line Losses

for m in range(branches):
    p=FromBus[m]-1
    q=ToBus[m]-1
    Losses[m] = Sij[p][q] + Sij[q][p]
    a = Losses[m]
    Lpij[m] = np.real(a[0])
    Lqij[m] = np.imag(a[0])




#Bus Power Injection

for m in range(bus):
    for n in range(bus):
        Si[m] = Si[m] + np.conj(V[m])*V[n]*Ybus[m][n]*basemva

PPi[:] = np.real(Si[:])
QQi[:] = -1*np.imag(Si[:])

#From Bus Injection
for i in range(branches):
    p = FromBus[i]-1
    q = ToBus[i]-1
    Pijfb[i] = np.real(Sij[p][q])
    Qijfb[i] = np.imag(Sij[p][q])

#To Bus Injection
for i in range(branches):
    p = FromBus[i]-1
    q = ToBus[i]-1
    Pijtb[i] = np.real(Sij[q][p])
    Qijtb[i] = np.imag(Sij[q][p])

Pg = PPi[:] + 100*(np.reshape(np.asarray(Ploads),(len(PPi),1)))
Qg = QQi[:] + 100*(np.reshape(np.asarray(Qloads),(len(QQi),1)))
buses = np.reshape(range(bus),(bus,1))
FromBus = np.reshape(FromBus,(branches,1))
ToBus = np.reshape(ToBus,(branches,1))
Ploads = 100*(np.reshape(np.asarray(Ploads),(len(PPi),1)))
Qloads = 100*(np.reshape(np.asarray(Qloads),(len(QQi),1)))

#header = ['Bus','Volt(p.u)','Angle(Deg)','P(Inj)','Q(Inj)','P(Gen)','Q(Gen)','P(Load)',  'Q(Load)']
np.savetxt("LoadFlowAnalysis.txt",np.column_stack((buses,Vmag, Vang, PPi, QQi, Pg, Qg, Ploads, Qloads)), delimiter=",", fmt='%d %.3f %.4f %.3f %.3f %.3f %.3f %.3f %.3f') #, newline = '\n', header = header)
np.savetxt("LineFlowAnalysis.txt",np.column_stack((FromBus, ToBus, Pijfb, Qijfb, Pijtb, Qijtb, Lpij, Lqij)), delimiter=",", fmt='%d %d %.3f %.3f %.3f %.3f %.3f %.3f')



def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt=None):
    """Solves the power flow using a fast decoupled method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, the FDPF matrices B prime
    and B double prime, and column vectors with the lists of bus indices
    for the swing bus, PV buses, and PQ buses, respectively. The bus voltage
    vector contains the set point for generator (including ref bus)
    buses, and the reference angle of the swing bus, as well as an initial
    guess for remaining magnitudes and angles. C{ppopt} is a PYPOWER options
    vector which can be used to set the termination tolerance, maximum
    number of iterations, and output options (see L{ppoption} for details).
    Uses default options if this parameter is not given. Returns the
    final complex voltages, a flag which indicates whether it converged
    or not, and the number of iterations performed.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    if ppopt is None:
        ppopt = ppoption()

    ## options
    tol     = ppopt['PF_TOL']
    max_it  = ppopt['PF_MAX_IT_FD']
    verbose = ppopt['VERBOSE']

    ## initialize
    converged = 0
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    ## set up indexing for updating V
    #npv = len(pv)
    #npq = len(pq)
    pvpq = r_[pv, pq]

    ## evaluate initial mismatch
    mis = (V * conj(Ybus * V) - Sbus) / Vm
    P = mis[pvpq].real
    Q = mis[pq].imag

    ## check tolerance
    normP = linalg.norm(P, Inf)
    normQ = linalg.norm(Q, Inf)
    if verbose > 1:
        sys.stdout.write('\niteration     max mismatch (p.u.)  ')
        sys.stdout.write('\ntype   #        P            Q     ')
        sys.stdout.write('\n---- ----  -----------  -----------')
        sys.stdout.write('\n  -  %3d   %10.3e   %10.3e' % (i, normP, normQ))
    if normP < tol and normQ < tol:
        converged = 1
        if verbose > 1:
            sys.stdout.write('\nConverged!\n')

    ## reduce B matrices
    Bp = Bp[array([pvpq]).T, pvpq].tocsc() # splu requires a CSC matrix
    Bpp = Bpp[array([pq]).T, pq].tocsc()

    ## factor B matrices
    Bp_solver = splu(Bp)
    Bpp_solver = splu(Bpp)

    ## do P and Q iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        ##-----  do P iteration, update Va  -----
        dVa = -Bp_solver.solve(P)

        ## update voltage
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * exp(1j * Va)

        ## evalute mismatch
        mis = (V * conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        ## check tolerance
        normP = linalg.norm(P, Inf)
        normQ = linalg.norm(Q, Inf)
        if verbose > 1:
            sys.stdout.write("\n  %s  %3d   %10.3e   %10.3e" %
                             (type,i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = 1
            if verbose:
                sys.stdout.write('\nFast-decoupled power flow converged in %d '
                    'P-iterations and %d Q-iterations.\n' % (i, i - 1))
            break

        ##-----  do Q iteration, update Vm  -----
        dVm = -Bpp_solver.solve(Q)

        ## update voltage
        Vm[pq] = Vm[pq] + dVm
        V = Vm * exp(1j * Va)

        ## evalute mismatch
        mis = (V * conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        ## check tolerance
        normP = linalg.norm(P, Inf)
        normQ = linalg.norm(Q, Inf)
        if verbose > 1:
            sys.stdout.write('\n  Q  %3d   %10.3e   %10.3e' % (i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = 1
            if verbose:
                sys.stdout.write('\nFast-decoupled power flow converged in %d '
                    'P-iterations and %d Q-iterations.\n' % (i, i))
            break

    if verbose:
        if not converged:
            sys.stdout.write('\nFast-decoupled power flow did not converge in '
                             '%d iterations.' % i)

    return V, converged, i
