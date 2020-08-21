import uproot, awkward
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from astropy.modeling import models, fitting
import pickle
import glob
import os


def MakePickles(rootfilename):

    # Open the root file and uproot the data into a dataframe.
    file = uproot.open(rootfilename)
    treename = "triplealpha_run" + rootfilename[-7:-5]
    tree = file[treename]

    # The code below will be almost identical to the calibration notebooks:

    brancharr = []
    df = pd.DataFrame()
    dfc = pd.DataFrame()

    # First, make an array of tree branch names like "b001_adc08..."
    chanlast = 0
    for adcnum in range(6):
        # first adc is adc08
        adcnum = adcnum + 8
        # print("adc: " + str(adcnum))
        if adcnum < 10:
            endname = "_adc0"
        else:
            endname = "_adc"
        if adcnum < 13:
            for chnum in range(32):
                chan = chnum + 1 + chanlast
                # Need to add leading 0's to the channel number
                if chan < 10:
                    startname = "b00"
                elif chan < 100:
                    startname = "b0"
                else:
                    startname = "b"
                branchname = startname + str(chan) + endname + str(adcnum)

                brancharr.append(branchname)
        if adcnum == 13:
            for chan in range(2):
                chan = chan + 1 + chanlast
                branchname = "b" + str(chan) + "_adc" + str(adcnum)
                brancharr.append(branchname)
        chanlast = chan

    # Now that all the channel names have been made, we can map them to the correct string to name the dataframe columns

    for i in range(len(brancharr)):
        branch = brancharr[i]
        if i < 32:
            if i < 16:
                df["U1_" + str(i) + "_O"] = tree.array(branch)
            if i > 15:
                df["U1_" + str(i - 16) + "_I"] = tree.array(branch)
        elif 31 < i < 64:
            if i - 32 < 16:
                df["D1_" + str(i - 32) + "_O"] = tree.array(branch)
            if i - 32 > 15:
                df["D1_" + str(i - 32 - 16) + "_I"] = tree.array(branch)
        elif 63 < i < 96:
            if i - 64 < 16:
                df["R1_" + str(i - 64) + "_O"] = tree.array(branch)
            if i - 64 > 15:
                df["R1_" + str(i - 64 - 16) + "_I"] = tree.array(branch)
        elif 95 < i < 128:
            if i - 96 < 16:
                df["L1_" + str(i - 96) + "_O"] = tree.array(branch)
            if i - 96 > 15:
                df["L1_" + str(i - 96 - 16) + "_I"] = tree.array(branch)
        elif 127 < i < 144:
            df["U2_" + str(i - 128)] = tree.array(branch)
        elif 143 < i < 160:
            df["D2_" + str(i - 144)] = tree.array(branch)
        elif i == 160:
            df["R2_0"] = tree.array(branch)
        elif i == 161:
            df["L2_0"] = tree.array(branch)

    # Now, the data frame has been created with all of the raw data, so we can open the pickle files we need to
    # gain match and calibrate the detectors.

    # dE gain match and calibrations:

    # First, we need to gain match the strip ends, which is done with a text file:
    gainparms = np.genfromtxt("textfiles/GainMatch.txt")
    # split the gainparms by detector:
    gaindet = []
    gaindet.append(gainparms[:16])
    gaindet.append(gainparms[16:32])
    gaindet.append(gainparms[32:48])
    gaindet.append(gainparms[48:])

    detnames = ["U1", "D1", "R1", "L1"]
    for i in range(4):
        for j in range(16):
            df[detnames[i] + "_" + str(j)] = (df[detnames[i] + "_" + str(j) + "_I"] - gaindet[i][j][0]) * \
                                             gaindet[i][j][2] + df[detnames[i] + "_" + str(j) + "_O"] - gaindet[i][j][1]

    # Open the pickle file for initial dE calibration:
    with open('dEcal.pkl', 'rb') as f:
        decal = pickle.load(f)

    for det in range(4):
        for strip in range(16):
            df[detnames[det] + "_" + str(strip) + "_E"] = decal[det][strip] * df[detnames[det] + "_" + str(strip)]

    # Need to fix the "smiles" in the next step:
    with open('SmileFits.pkl', 'rb') as f:
        smfits = pickle.load(f)

    for det in range(4):
        for strip in range(16):
            df[detnames[det] + "_" + str(strip) + "_upos"] = (df[detnames[det] + "_" + str(strip) + "_O"] -
                                                              df[detnames[det] + "_" + str(strip) + "_I"]) / \
                                                             (df[detnames[det] + "_" + str(strip) + "_O"] +
                                                              df[detnames[det] + "_" + str(strip) + "_I"])

            fit = smfits[det][strip]
            smfix = 5.8 / (df[detnames[det] + "_" + str(strip) + "_upos"] ** 2 * fit[0] + df[
                detnames[det] + "_" + str(strip) + "_upos"] * fit[1] + fit[2])

            df[detnames[det] + "_" + str(strip) + "_E"] = df[detnames[det] + "_" + str(strip) + "_E"] * smfix

    # Now that we've fixed the smiles we need to correct the dE energies
    with open('dEcorr.pkl', 'rb') as f:
        decorr = pickle.load(f)

    for det in range(4):
        for strip in range(16):
            dfc[detnames[det] + "_" + str(strip) + "_E"] = df[detnames[det] + "_" + str(strip) + "_E"] * \
                                                          decorr[det][strip]

    # Now that they are successfully calibrated, the final step is to calibrate the dE detector strips for position.
    # for now, this calibration just makes the position go from 0 to 1.

    with open('poscal.pkl', 'rb') as f:
        poscal = pickle.load(f)

    for det in range(4):
        for strip in range(16):
            edgecf = poscal[det][strip][0]
            edgecl = poscal[det][strip][0]

            # We need this try except here because one of the strips wasn't working and the edges are 0...
            try:
                dfc[detnames[det] + "_" + str(strip) + "_pos"] = (df[detnames[det] + "_" + str(strip) + "_upos"] + (
                            -1 * edgecf)) * (1 / (edgecl - edgecf))
            except ZeroDivisionError:
                dfc[detnames[det] + "_" + str(strip) + "_pos"] = df[detnames[det] + "_" + str(strip) + "_upos"]

    # Now that the dE detector calibration is out of the way, we can go ahead and calibrate the E detector strips!
    edetnames = ["U2", "D2", "R2", "L2"]
    # Keep in mind that R2 and L2 only have one strip!
    with open('Ecal.pkl', 'rb') as f:
        ecal = pickle.load(f)

    for det in range(4):
        if det < 2:
            numstrips = 16
        if det > 1:
            numstrips = 1
        for strip in range(numstrips):
            dfc[edetnames[det] + "_" + str(strip) + "_E"] = ecal[det][strip] * df[edetnames[det] + "_" + str(strip)]

    # Now, the final step is to make our run pickle files so we can open these dataframes later.

    #dfc.to_pickle("../runpickles/run" + rootfilename[-7:-5] + ".pkl")
    #dfc.to_hdf("../runpickles/run" + rootfilename[-7:-5] + ".h5", key='dfc', mode='w')
    dfc.to_parquet("../runpickles/run" + rootfilename[-7:-5] + ".parquet.gzip", compression='gzip')
    print("run" + rootfilename[-7:-5] + ".pkl was generated!")


if __name__ == "__main__":
    print("Making pickle files from ROOT!")

    # First, get the list of root file names:
    rootdir = "../rootfiles/"

    list_rootfiles = glob.glob(rootdir + 'run*.root')

    # Loop through the root files to
    for k in range(len(list_rootfiles)):

        MakePickles(list_rootfiles[k])

        #print(list_rootfiles[i][-7:-5])

