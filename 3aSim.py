# 11B(d,n) simulation code:
import numpy as np
import pandas as pd
import seaborn as sbn
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/mhall12/SOLSTISE_Codes/')
from stopyt import desorb

def makeevts(numevts, beamenum, telepos, tgtthk):

    df = pd.DataFrame()

    # Since the masses are constant, we'll just define them here...
    # masses are 11B (0), d (1), n (2), 12C (3), 4He (4), 8Be (5)
    mass = [10255.1, 1876.12, 939.565, 11177.9, 3728.4, 7456.89]

    # define a numpy array of the beam energy with the correct number of events:
    beame = np.zeros(numevts) + beamenum
    zbeam = np.zeros(numevts) + 5
    abeam = np.zeros(numevts) + 11

    # Assume a reaction depth and calculate the beam energy after loss:
    depth = np.random.rand(numevts)

    zt = [6, 1]
    at = [12, 2]
    num = [1, 2]
    density = 0.94
    gas = False
    thickness = tgtthk * depth

    dfout = desorb(zbeam, abeam, beame, zt, at, num, gas, density, thickness, 0, 0, beame)
    beame = dfout['Energy_i'].to_numpy() - dfout['DeltaE_tot'].to_numpy()

    print(beame)

    # define the 12C energy levels that can be populated, some are not listed on NNDC, not sure where
    # Kelly got these from:
    exc = [7.6542, 9.641, 10.3, 10.84, 11.16, 11.83, 12.71, 13.35]

    # reaction and decay q-values:
    qval = mass[0] + mass[1] - mass[2] - mass[3]
    qdec1 = mass[3] - mass[4] - mass[5]
    qdec2 = mass[5] - mass[4]*2

    # First step is the 11B(d,n) reaction, and we must randomly get one of the 12C excited states:

    levnums = np.random.randint(len(exc), size=numevts)
    exenergy = np.zeros_like(beame)

    for i in range(len(exc)):
        exmask = levnums == i
        exenergy[exmask] = exc[i]




if __name__=="__main__":
    print("This code does something")

    numevents = int(input("Enter the number of events you would like to simulate: "))
    beamenergy = float(input("Enter the beam energy in MeV: "))

    telescopepos = int(input("If the L/R detector radius is 6.5 cm (3 cm), enter 1 (2): "))

    tgtthick = float(input("Enter the CD2 target thickness in mg/cm2 (180 or 186): "))

    df = makeevts(numevents, beamenergy, telescopepos, tgtthick)

    # now save the df to a pickle or gzip: