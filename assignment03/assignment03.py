
# Assignment 03

import numpy as np
import scipy.signal 
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt




def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    
    t = np.arange(0,length_secs,1.0/sampling_rate_Hz)
    x = amplitude*np.sin((2*np.pi * frequency_Hz * t) + phase_radians)
    
    return (t,x)


def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    
    sq = 0
    for i in range(1,20,2):
        sq += generateSinusoidal(amplitude, sampling_rate_Hz, i*frequency_Hz, length_secs, phase_radians)[1] / i
        sq_final = (4/np.pi)* sq         
        t = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)[0]
    return (t,sq)
   
def computeSpectrum(x, sampling_rate_Hz):
    
    hN = (x.size//2)+1
    X = np.fft.fft(x)                                     
    
    Xabs = abs(X[:hN])       
    Xphase = np.angle(X[:hN])

    Xre = np.real(X[:hN])
    XIm = np.imag(X[:hN])        
    f = np.linspace(0, sampling_rate_Hz, len(X[:hN]))
    
    return (f, Xabs, Xphase, Xre, XIm)

def generateBlocks(x, sampling_rate_Hz, block_size, hop_size):
    
    if block_size < hop_size:
        print("Error in block and hop sizes")
        return
    
    
    hM1 = (block_size+1)//2                                 
    hM2 = block_size//2   
    x = np.append(np.zeros(hM2),x)                 
    x = np.append(x,np.zeros(hM1))                
    pin = hM1                                            
    pend = x.size-hM1 

    s = []
    X = []

    while pin<=pend:  
        s1 = pin
        s.append(s1)
        x1 = x[pin-hM1:pin+hM2]
        X.append(x1) 

        pin += hop_size

    s = np.array(s)
    X = np.array(X)
    t = s/sampling_rate_Hz
    
    return t,X
                    
                    
def mySpecgram(x, block_size, hop_size, sampling_rate_Hz, window_type):
    
    t,x_block = generateBlocks(x,sampling_rate_Hz,block_size,hop_size)
    
    xMx = []
     
    if (window_type == 'rect'):
        w = scipy.signal.boxcar(block_size)
    elif(window_type == 'hann'):
        w = np.hanning(block_size)
    
    
    for row in range(x_block.shape[0]):
    
        xw = np.multiply(x_block[row], w)
        f,mX = computeSpectrum(xw , sampling_rate_Hz)[:2]
        xMx.append(mX)
        
    xMx = np.array(xMx)
    f_vector = np.linspace(0,len(x),block_size)
     
    return t,f_vector,xMx 


def plotSpecgram(freq_vector, time_vector, magnitude_spectrogram, title, xlabel, ylabel):
    if len(freq_vector) < 2 or len(time_vector) < 2:
        return

    Z = 20. * np.log10(magnitude_spectrogram)
    Z = np.flipud(Z)
  
    pad_xextent = (time_vector[1] - time_vector[0]) / 2
    xmin = np.min(time_vector) - pad_xextent
    xmax = np.max(time_vector) + pad_xextent
    extent = xmin, xmax, freq_vector[0], freq_vector[-1]
  
    im = plt.imshow(Z, None, extent=extent, 
                           origin='upper')
    plt.axis('auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
    
def plotFn(x,y,title,axis,xlabel,ylabel):
    plt.figure()
    plt.title(title)
    plt.plot(x,y)
    plt.axis(axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    

def main():
    
    #Question_1
    
    #Setting values to the sinusoid
    
    amplitude = 1.0    
    frequency_Hz = 400
    phase_radians = np.pi/2
    sampling_rate_Hz = 44100
    length_secs = 0.5
    
    (t,x) = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians) 
    plotFn(t[:220],x[:220],'Sinusoidal wave', [0,0.005,-2,2],'Time(sec)', 'Amplitude')
    
    #Question_2
    
    phase_radians = 0
    (t1,x1) = generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians = 0) #Approximate a Square Wave
    plotFn(t[:220],x1[:220],'Square wave', [0,0.005,-2,2],'Time','Amplitude')         #Plotting the Square wave 
   

    # Question 3
    (f ,Xabs, Xphase, Xre, XIm) = computeSpectrum(x,sampling_rate_Hz) #compute spectrum of Sinusoidal wave
    (f1 ,Xabs1, Xphase1, Xre1, XIm1) = computeSpectrum(x1,sampling_rate_Hz) # Compute spectrum of a square wave
       
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Magnitude Spectrum - Sinusoidal')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.plot(f,Xabs)
    
    plt.subplot(2,1,2)
    plt.plot(f, Xphase)
    plt.title('Phase Spectrum - Sinusoidal')
    plt.xlabel('Frequency')
    plt.ylabel('Phase') 
    plt.show()
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(f1,Xabs1)
    plt.title('Magnitude Spectrum - Square wave')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    
    
    plt.subplot(2,1,2)
    plt.plot(f1, Xphase1)
    plt.title('Phase Spectrum - Square Wave')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')  
    plt.show()  
    
    #Question_4
    
    block_size = 2048
    hop_size = 1024
    window_type1 = 'hann'
    
    #Rect window spectrogram
    window_type2 = 'rect'
    t_array,f_vector, mag = mySpecgram(x1, block_size, hop_size, sampling_rate_Hz, window_type2)
    plotSpecgram(f_vector, t_array, np.transpose(mag), title = 'Spectrogram of Square wave - Rect Window', xlabel='Time(s)', ylabel = 'Frequency(Hz)')
    
    #Hanning Window Spectrogram
    t_array,f_vector, mag = mySpecgram(x1, block_size, hop_size, sampling_rate_Hz, window_type1)
    plotSpecgram(f_vector, t_array, np.transpose(mag), title = 'Spectrogram of Square wave - Hanning Window', xlabel='Time(s)', ylabel = 'Frequency(Hz)')
    
    #Pxx, freqs, bins, im  = plt.specgram(x1, NFFT=block_size, Fs=sampling_rate_Hz, noverlap=hop_size)
    
    
if __name__ == '__main__':
    main()
    
    



