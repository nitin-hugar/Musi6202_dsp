

import numpy as np
from scipy.io.wavfile import read
import scipy.signal
import matplotlib.pyplot as plt

#function to import the soundfiles and extract only the left channel from a stereo wave file. 
def loadSoundFile(filename):
    fs,x = scipy.io.wavfile.read(filename)
    return np.array(x[:,0], dtype = float)


#function to perform cross correlation of signals    
def crossCorr(x,y):
    return scipy.signal.correlate(x,y)


#function to find the height of the peaks
def roundup(x):
         return int(np.floor(x / 1e+11)) * 1e+11

#function to find the snare positions in the drum loop
def findSnarePosition(snareFilename, drumloopFilename):
    x = loadSoundFile(snareFilename)
    y = loadSoundFile(drumloopFilename)
    z = crossCorr(x,y)
    return scipy.signal.find_peaks(z, height= roundup(np.max(z)))[0]


#main function calls all the other functions    
def main():
    snareFilename = 'snare.wav'
    drumloopFilename = 'drum_loop.wav'
    x = loadSoundFile(snareFilename)
    y = loadSoundFile(drumloopFilename)
    z = crossCorr(x,y)
    pos = findSnarePosition(snareFilename,drumloopFilename)
    np.savetxt('results/02-snareLocation.txt', pos) #exports the snare positions as a txt file
    
#Plot x,y and z
    plt.figure()
    
    plt.subplot(3,1,1)
    plt.title('Snare')
    plt.plot(x)
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    
    plt.subplot(3,1,2)
    plt.title('Drum Loop')
    plt.plot(y)
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.subplot(3,1,3)
    
    plt.title('Correlated Signal')
    plt.plot(z)
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    
    plt.savefig('results/01-correlation.png') #exports the correlation graphs as a png file

if __name__ == "__main__":
    main()    



