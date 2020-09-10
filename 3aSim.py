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
    # Also need the thickness of the target that the 12C exits through
    thickness_back = tgtthk - thickness

    dfout = desorb(zbeam, abeam, beame, zt, at, num, gas, density, thickness, 0, 0, beame)
    beame = dfout['Energy_i'].to_numpy() - dfout['DeltaE_tot'].to_numpy()

    beamstrag = dfout['E_strag_FWHM'].to_numpy()

    # Beame uses the beam straggling
    beame = np.random.normal(beame, beamstrag)

    # We'll define the reaction to be in the z-direction, so we can set the momentum:
    p0z = np.sqrt(2 * mass[0] * beame)

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

    # Now, each event will have a different populated excitation energy
    for i in range(len(exc)):
        exmask = levnums == i
        exenergy[exmask] = exc[i]

    # First step is now to calculate the range of angles that the 12C can come out:
    # Total Energy:
    etot = beame + qval - exenergy
    qq = (mass[0] + mass[1] - mass[2] - mass[3]) - exenergy

    # random number for theta3:
    ntheta = np.random.rand(numevts) * np.pi

    energy_n = ((np.sqrt(mass[0] * mass[2] * beame) * np.cos(ntheta) +
                np.sqrt(mass[0] * mass[2] * beame * np.cos(ntheta) ** 2 +
                        (mass[3] + mass[2]) * (mass[3] * qq + (mass[3] - mass[0]) *
                                               beame))) / (mass[3] + mass[2])) ** 2


    # Calculate the 12C kinetic energy from the total momentum conservation:
    energy_c = qval - energy_n - exenergy + beame

    # Calculate theta 3 from the momentum conservation:
    p2 = np.sqrt(2 * mass[2] * energy_n)
    p3 = np.sqrt(2 * mass[3] * energy_c)

    # Get the angle of the carbon 12 using conservation of momentum in the z-direction
    ctheta = np.arccos((p0z - p2*np.cos(ntheta)) / p3)
    cphi = np.random.rand(numevts) * np.pi * 2

    # Now that the initial reaction is simulated, we can start simulating the decays:
    # first is the breakup of 12C into an alpha (4) and 8Be (5)

    betheta = np.random.rand(numevents) * np.pi
    bephi = np.random.rand(numevents) * 2 * np.pi
    betheta_tot = ctheta + betheta
    bephi_tot = cphi + bephi

    betheta_tot = np.where(betheta_tot > np.pi, 2*np.pi - betheta_tot, betheta_tot)
    bephi_tot = np.where(bephi_tot > (2 * np.pi), bephi_tot - 2 * np.pi, bephi_tot)

    a1theta_tot = np.pi - betheta_tot
    a1phi_tot = bephi_tot + np.pi

    a1phi_tot = np.where(a1phi_tot > (2 * np.pi), a1phi_tot - 2 * np.pi, a1phi_tot)

    # Now, with the angles defined we can figure out the energies of these particles using conservation of momentum:
    p3x = p3 * np.sin(ctheta) * np.cos(cphi)
    p3y = p3 * np.sin(ctheta) * np.sin(cphi)
    p3z = p3 * np.cos(ctheta)

    Ecmdec1 = exenergy + qdec1
    mred1 = mass[4] * mass[5] / (mass[4] + mass[5])
    pdec1 = np.sqrt(2 * mred1 * Ecmdec1)

    pbex = pdec1 * np.sin(betheta_tot) * np.cos(bephi_tot) + mass[4] / mass[3] * p3x
    pbey = pdec1 * np.sin(betheta_tot) * np.sin(bephi_tot) + mass[4] / mass[3] * p3y
    pbez = pdec1 * np.cos(betheta_tot) + mass[4] / mass[3] * p3z

    pa1x = pdec1 * np.sin(a1theta_tot) * np.cos(a1phi_tot) + mass[5] / mass[3] * p3x
    pa1y = pdec1 * np.sin(a1theta_tot) * np.sin(a1phi_tot) + mass[5] / mass[3] * p3y
    pa1z = pdec1 * np.cos(a1theta_tot) + mass[5] / mass[3] * p3z

    ea1 = (pa1x**2 + pa1y**2 + pa1z**2) / (2 * mass[4])

    # Now, the Be has a chance to decay:
    a2theta = np.random.rand(numevents) * np.pi
    a2phi = np.random.rand(numevents) * 2 * np.pi

    a2theta_tot = a2theta + betheta_tot
    a2theta_tot = np.where(a2theta_tot > np.pi, 2*np.pi - a2theta_tot, a2theta_tot)

    a2phi_tot = bephi_tot + a2phi
    a2phi_tot = np.where(a2phi_tot > (2 * np.pi), a2phi_tot - np.pi * 2, a2phi_tot)

    a3theta_tot = np.pi - a2theta_tot
    a3phi_tot = a2phi_tot + np.pi

    a3phi_tot = np.where(a3phi_tot > (2 * np.pi), a3phi_tot - 2 * np.pi, a3phi_tot)

    Ecmdec2 = qdec2

    mred2 = mass[4] * mass[4] / (mass[4] + mass[4])
    pdec2 = np.sqrt(2 * mred2 * Ecmdec2)

    pa2x = pdec2 * np.sin(a2theta_tot) * np.cos(a2phi_tot) + mass[4]/mass[5] * pbex
    pa2y = pdec2 * np.sin(a2theta_tot) * np.sin(a2phi_tot) + mass[4]/mass[5] * pbey
    pa2z = pdec2 * np.cos(a2theta_tot) + mass[4] / mass[5] * pbez

    pa3x = pdec2 * np.sin(a3theta_tot) * np.cos(a3phi_tot) + mass[4]/mass[5] * pbex
    pa3y = pdec2 * np.sin(a3theta_tot) * np.sin(a3phi_tot) + mass[4]/mass[5] * pbey
    pa3z = pdec2 * np.cos(a3theta_tot) + mass[4] / mass[5] * pbez


if __name__=="__main__":
    print("This code does something")

    try:
        numevents = int(input("Enter the number of events you would like to simulate: "))
    except ValueError:
        numevents = 100

    try:
        beamenergy = float(input("Enter the beam energy in MeV: "))
    except ValueError:
        beamenergy = 18

    try:
        telescopepos = int(input("If the L/R detector radius is 6.5 cm (3 cm), enter 1 (2): "))
    except ValueError:
        telescopepos = 1

    try:
        tgtthick = float(input("Enter the CD2 target thickness in ug/cm2 (180 or 186): "))
        thtthick = tgtthick / 1000
    except ValueError:
        tgtthick = 0.180

    df = makeevts(numevents, beamenergy, telescopepos, tgtthick)

    # now save the df to a pickle or gzip: