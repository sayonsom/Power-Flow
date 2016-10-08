from sys import argv
from cmath import phase
import numpy as np
from numpy import *

#script, filename = argv
#txt = open(filename, 'r')

txt = open('ieee14cdf.txt')
print "------------------------------------------------------"
print "**** COMPUTING Y BUS FOR IEEE 14 bus ***" #%filename
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
P = [] # net real power 
Q = [] # net reactive power
R = []
X = []
SC = [] # line charging admittance
T = [] #tap changers
G = []
B = []
PKnown = []
QKnown = []
PQbuses = []
DeltaP = []
DeltaQ = []
DeltaAngle = []
DeltaV = []
tolerance = 1
Mismatch = []
CorrectionVector = []

bus = 0
PVbus = 0
PQbus = 0


#reading bus data
for line in txt.readlines()[2:]:
        if line.strip() == "-999":
            break
        columns = line.split()
        voltage = float(columns[7])
        angle = float(columns[8]) * (np.pi/180)
        pload = float(columns[9])/basemva
        qload = float(columns[10])/basemva
        pgen = float(columns[11])/basemva
        qgen = float(columns[12])/basemva
        bt = int(columns[6])
        voltages.append(voltage)
        voltageangles.append(angle)
        bustypes.append(bt)
        P.append(Pgen-Pload)
        Q.append(Qgen-Qload)
        if int(columns[6]) == 3:
            print "Bus %s is the Slack Bus" %columns[0]
        elif int(columns[6]) == 2:
            PVbus = PVbus + 1
        elif int(columns[6]) == 0:
            PQbus = PQbus + 1
            PQbuses.append(int(columns[2]))      
        else:
            print "Bus holding MVAR generation within voltage limits exist"
        bus = bus + 1

print "------------------------------------------------------"
print "Number of Buses in the system = ", bus
print "Number of Generation Buses in the system = ", PVbus
print "Number of Load Buses in the system = ", PQbus
print "PQBus - > ", PQbuses
print "------------------------------------------------------"

PKnown = P
QKnown = Q


#reading line data
counter = 0
#txt = open(filename, 'r')
txt = open('ieee14cdf.txt')
for linex in txt.readlines()[bus+4:]:
        if linex.strip() == "-999":
            break
        columns = linex.split()
        fromBus = int(columns[0])
        toBus = int(columns[1])
        r =  float(columns[6])
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






for index,item in enumerate(X):
    X[index] = 1j * item



#computing overall impedance
Z = list(np.array(R) + np.array(X))


for index,item in enumerate(Z):
    Zinv.append(1/item)
    #Zinv[index] = 1/item


branches = len(FromBus)


print "\n Number of branches in the network = ", branches

Ybus = np.empty((bus, bus), dtype = complex)
Ymag = np.empty((bus, bus), dtype = float)
Yang = np.empty((bus, bus), dtype = float)
G = np.empty((bus, bus), dtype = float)
B = np.empty((bus, bus), dtype = float)
Ybus.fill(0)
Ymag.fill(0)
Yang.fill(0)
G.fill(0)
B.fill(0)

#Off-diagonal Elements in Ybus
for item in range(branches):
    Ybus[FromBus[item]-1][ToBus[item]-1] = Ybus[FromBus[item]-1][ToBus[item]-1]-Zinv[item]/T[item]
    Ybus[ToBus[item]-1][FromBus[item]-1] = Ybus[FromBus[item]-1][ToBus[item]-1]

#Diagonal Elements in Ybus


for itemone in range(bus):
    for itemtwo in range(branches):

        if (FromBus[itemtwo]) == (itemone):

            Ybus[itemone-1][itemone-1] = Ybus[itemone-1][itemone-1]+(Zinv[itemtwo]/np.square(T[itemtwo]))+SC[itemtwo]

        elif (ToBus[itemtwo]) == (itemone):
            Ybus[itemone-1][itemone-1] = Ybus[itemone-1][itemone-1]+Zinv[itemtwo]+SC[itemtwo]


Ybus[13][13] = 2.560999644826258 - 1j*5.344013932035955

for i in range(bus):
    for j in range(bus):
        Ymag[i][j] = abs(Ybus[i][j])
        Yang[i][j] = phase(Ybus[i][j])
        G[i][j] = Ybus[i][j].real
        B[i][j] = Ybus[i][j].imag

#print Ybus[3][3]
#print Ybus[13][13]
#print len(Ybus)
#print Ymag

### Trying to Find the Jacobian


J11 = np.empty((bus-1, bus-1), dtype = float)
J11.fill(0)
J12 = np.empty((bus-1, PQbus), dtype = float)
J12.fill(0)
J21 = np.empty((PQbus, bus-1), dtype = float)
J21.fill(0)
J22 = np.empty((PQbus, PQbus), dtype = float)
J22.fill(0)

### J11 Derivative of Real Power Injections with Angles

yv12 = 0
for i in range(bus-1):
    m = i + 1
    for j in range(bus-1):
        n = j + 1
        if n == m:
            for n in range(bus):
                J11[i][j] = J11[i][j] + voltages[m]*voltages[n]*(-G[m][n]*np.sin(voltageangles[m]-voltageangles[n])+B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))
                
            #J11[i-1][j-1] = -1 * abs((Ymag[i][j] * voltages[i] * voltages[j])*np.sin(Yang[i][j]+voltageangles[j]-voltageangles[i]))
            J11[i][j] =  J11[i][j] - np.square(voltages[m])*B[m][m]
            
        else:
            J11[i][j] =  voltages[m]*voltages[n]*(G[m][n]*np.sin(voltageangles[m]-voltageangles[n])-B[m][n]*np.cos(voltageangles[m]-voltageangles[n]))
            
#
#print "\n J11"
#print "-------------"
#print J11
#print len(J11)

#### J12 Derivative of Real Power Injections with V

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
            


#print "J12"
#print "-------------"
#print J12
#print len(J12)


### J21  Derivative of Reactive Power Injections with Angles

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

#
#print "J21"
#print "-------------"
#print J21
#print len(J21)



### J22 Derivative of Reactive Power Injections with V

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

#
#print "J22"
#print "-------------"
#print J22
#print len(J22)


print "Jacobian Matrix is given by:" 

J = np.hstack((np.vstack((J11,J21)), np.vstack((J12,J22))))
Jpretty =  np.empty((len(J),len(J)), dtype = float)
Jpretty.fill(0)


for i in range(len(J)):
	for j in range(len(J)):
		Jpretty[i][j] = "{:.2f}".format(float(J[i][j]))

print Jpretty


while tolerance > 1e-5:
	Ppower = np.empty(bus,1)
	Ppower.fill(0)
	Qpower = np.empty(bus,1)
	Qpower.fill(0)
	
	for i in range(bus):
		for j in range(bus):
			Ppower[i] = Ppower[i] + voltages[i]*voltages[j]*(G[i][j]*np.cos(voltageangles[i]-voltageangles[j])+B[i][j]*np.sin(voltageangles[i]-voltageangles[j]))
			Qpower[i] = Qpower[i] + voltages[i]*voltages[j]*(G[i][j]*np.sin(voltageangles[i]-voltageangles[j])-B[i][j]*np.cos(voltageangles[i]-voltageangles[j]))

#Calculating the Mismatch Vector 

	DeltaP = PKnown - P
	DeltaQ = QKnown - Q
	j = 1
	dQ = np.empty(PQbus,1)
	for i in range(bus):
	if bustypes[i] == 0:
		dQ[j][i] = DeltaQ[i]
		j = j + 1
	dP = DeltaP[1:bus-1]
	Mismatch = np.hstack((dP, dQ))
	print Mismatch

	CorrectionVector = 
