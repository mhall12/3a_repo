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
    mass = [10255.1031, 1875.612928, 939.5654133, 11178.0, 3728.40133, 7456.89449]

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

    if telepos == 1:
        lrrad = 6.5
    else:
        lrrad = 3

    zde = 17.78

    dfout = desorb(zbeam, abeam, beame, zt, at, num, gas, density, thickness, 0, 0, beame)
    beame = dfout['Energy_i'].to_numpy() - dfout['DeltaE_tot'].to_numpy()

    beamstrag = dfout['E_strag_FWHM'].to_numpy()

    # Beame uses the beam straggling
    beame = np.random.normal(beame, beamstrag)

    # We'll define the reaction to be in the z-direction, so we can set the momentum:
    p0z = np.sqrt(2 * mass[0] * beame)

    # define the 12C energy levels that can be populated, some are not listed on NNDC, not sure where
    # Kelly got these from:
    exc = [7.6542]#, 9.641, 10.3, 10.84, 11.16, 11.83, 12.71, 13.35]

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

    nphi = cphi + np.pi
    nphi = np.where(nphi > (2*np.pi), nphi - np.pi*2, nphi)

    npx = p2 * np.sin(ntheta) * np.cos(nphi)
    npy = p2 * np.sin(ntheta) * np.sin(nphi)
    npz = p2 * np.cos(ntheta)
    cpx = p3 * np.sin(ctheta) * np.cos(cphi)

    # Now that the initial reaction is simulated, we can start simulating the decays:
    # first is the breakup of 12C into an alpha (4) and 8Be (5)

    # This detector setup isn't very efficient, so we'll force theta to be close to the value we want.
    thrange = np.arctan((lrrad+5)/zde) - np.arctan(lrrad/zde)
    thmax = np.pi - np.arctan(lrrad/zde)

    #betheta = np.random.rand(numevents) * np.pi
    betheta = thmax - np.random.rand(numevents) * thrange
    bephi = np.random.rand(numevents) * 2 * np.pi
    betheta_tot = ctheta + betheta
    bephi_tot = cphi + bephi

    #betheta_tot = np.where(betheta_tot > np.pi, 2*np.pi - betheta_tot, betheta_tot)
    bephi_tot = np.where(bephi_tot > (2 * np.pi), bephi_tot - 2 * np.pi, bephi_tot)

    a1theta_tot = np.pi - betheta_tot
    a1phi_tot = bephi_tot + np.pi

    a1phi_tot = np.where(a1phi_tot > (2 * np.pi), a1phi_tot - 2 * np.pi, a1phi_tot)

    print(a1phi_tot * 180/np.pi)

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

    ebe = (pbex**2 + pbey**2 + pbez**2) / (2 * mass[5])

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

    pxtot = pa2x + pa3x + pa1x + npx
    pytot = pa2y + pa3y + pa1y + npy
    pztot = pa2z + pa3z + pa1z + npz

    ea2 = (pa2x**2 + pa2y**2 + pa2z**2) / (2 * mass[4])
    ea3 = (pa3x**2 + pa3y**2 + pa3z**2) / (2 * mass[4])

    # Now that we have the alpha energy/angles, etc, we need to reduce their energy based on how far they travel
    # in the target:
    zalpha = np.zeros(numevts) + 2
    aalpha = np.zeros(numevts) + 4

    # need to calculate the target thickness that the alphas go through:
    thka1 = np.where(a1theta_tot > (np.pi/2), thickness / np.sin(a1theta_tot-np.pi/2),
                     thickness_back / np.cos(a1theta_tot))
    thka2 = np.where(a2theta_tot > (np.pi/2), thickness / np.sin(a2theta_tot-np.pi/2),
                     thickness_back / np.cos(a2theta_tot))
    thka3 = np.where(a3theta_tot > (np.pi/2), thickness / np.sin(a3theta_tot-np.pi/2),
                     thickness_back / np.cos(a3theta_tot))

    dfout = desorb(zalpha, aalpha, ea1, zt, at, num, gas, density, thka1, 0, 0, ea1)
    ea1 = ea1 - dfout['DeltaE_tot'].to_numpy()

    dfout = desorb(zalpha, aalpha, ea2, zt, at, num, gas, density, thka2, 0, 0, ea1)
    ea2 = ea2 - dfout['DeltaE_tot'].to_numpy()

    dfout = desorb(zalpha, aalpha, ea3, zt, at, num, gas, density, thka3, 0, 0, ea1)
    ea3 = ea3 - dfout['DeltaE_tot'].to_numpy()

    # Now, we need to figure out the detector positions and where all of these particles will be detected.
    # First, calculate r for each particle at z=17.78 cm, which is the distance from the target to the
    # detectors:

    ra1 = zde / np.cos(a1theta_tot)
    ra2 = zde / np.cos(a2theta_tot)
    ra3 = zde / np.cos(a3theta_tot)

    # Calculate x, y position for each alpha:

    xa1 = ra1 * np.sin(a1theta_tot) * np.cos(a1phi_tot)
    ya1 = ra1 * np.sin(a1theta_tot) * np.sin(a1phi_tot)

    #print(xa1, ya1)

    xa2 = ra2 * np.sin(a2theta_tot) * np.cos(a2phi_tot)
    ya2 = ra2 * np.sin(a2theta_tot) * np.sin(a2phi_tot)

    xa3 = ra3 * np.sin(a3theta_tot) * np.cos(a3phi_tot)
    ya3 = ra3 * np.sin(a3theta_tot) * np.sin(a3phi_tot)

    df["Alpha 1 Energy"] = ea1
    df["Alpha 2 Energy"] = ea2
    df["Alpha 3 Energy"] = ea3

    df["A1 Phi"] = a1phi_tot * 180/np.pi

    df["A1 Theta"] = a1theta_tot * 180/np.pi
    df["A2 Theta"] = a2theta_tot * 180/np.pi
    df["A3 Theta"] = a3theta_tot * 180/np.pi

    df["A1x"] = xa1
    df["A1y"] = ya1

    df["A2x"] = xa2
    df["A2y"] = ya2

    df["A3x"] = xa3
    df["A3y"] = ya3

    df["nn Theta"] = ntheta * 180/np.pi

    # det name of 4 means no detection.
    df["Det Name a1"] = np.zeros(numevts) + 4
    df["Det Name a2"] = np.zeros(numevts) + 4
    df["Det Name a3"] = np.zeros(numevts) + 4

    # Set the masks for each det here:
    for i in range(4):
        detnames = ["U", "D", "L", "R"]
        # Detx1 and detx2 are the limits for the detector in the x direction
        # Detectors are 5x5 cm.
        detx1 = [-2.5, -2.5, -1 * lrrad - 5, lrrad]
        detx2 = [2.5, 2.5, -1 * lrrad, lrrad + 5]
        dety1 = [6.5, -11.5, -2.5, -2.5]
        dety2 = [11.5, -6.5, 2.5, 2.5]
        detmask1 = (a1theta_tot < np.pi/2) & (xa1 > detx1[i]) & (xa1 < detx2[i]) & (ya1 > dety1[i]) & (ya1 < dety2[i])
        detmask2 = (a2theta_tot < np.pi/2) & (xa2 > detx1[i]) & (xa2 < detx2[i]) & (ya2 > dety1[i]) & (ya2 < dety2[i])
        detmask3 = (a3theta_tot < np.pi/2) & (xa3 > detx1[i]) & (xa3 < detx2[i]) & (ya3 > dety1[i]) & (ya3 < dety2[i])

        df["Det Name a1"][detmask1] = i
        df["Det Name a2"][detmask2] = i
        df["Det Name a3"][detmask3] = i

    detectedmask1 = df['Det Name a1'] < 4
    detectedmask2 = df['Det Name a2'] < 4
    detectedmask3 = df['Det Name a3'] < 4

    # Initialize the strip at 17, which doesn't correspond to a detector.
    df["Strip a1"] = np.zeros(numevts) + 17
    df["Strip a2"] = np.zeros(numevts) + 17
    df["Strip a3"] = np.zeros(numevts) + 17

    # Also need to assign strip numbers... each strip is 5/16 cm wide (0.3125)

    for i in range(4):
        for j in range(16):
            # if U/D the
            if i < 2:
                stripx1 = detx1[i] + j * 5/16
                stripx2 = stripx1 + 5/16

                stripy1 = dety1[i]
                stripy2 = dety2[i]
            else:
                stripx1 = detx1[i]
                stripx2 = detx2[i]

                stripy1 = dety1[i] + j * 5/16
                stripy2 = stripy1 + 5/16

            # Now we can use np.where to assign what strip was hit.
            stripmaska1 = (xa1 > stripx1) & (xa1 < stripx2) & (ya1 > stripy1) & (ya1 < stripy2) & (df["Det Name a1"] == i)
            stripmaska2 = (xa2 > stripx1) & (xa2 < stripx2) & (ya2 > stripy1) & (ya2 < stripy2) & (df["Det Name a2"] == i)
            stripmaska3 = (xa3 > stripx1) & (xa3 < stripx2) & (ya3 > stripy1) & (ya3 < stripy2) & (df["Det Name a3"] == i)

            df["Strip a1"] = np.where(stripmaska1, j, df["Strip a1"])
            df["Strip a2"] = np.where(stripmaska2, j, df["Strip a2"])
            df["Strip a3"] = np.where(stripmaska3, j, df["Strip a3"])

    print(df[(detectedmask1 & detectedmask2) | (detectedmask1 & detectedmask3) | (detectedmask2 & detectedmask3)])

    print(df)

if __name__=="__main__":

    print("This code does something")

    try:
        numevents = int(input("Enter the number of events you would like to simulate: "))
    except ValueError:
        numevents = 1

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