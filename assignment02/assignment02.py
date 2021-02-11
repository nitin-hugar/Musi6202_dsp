import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time
import scipy.io.wavfile as wav

#Plot x and h
#Function to perform convolution of signals

def myTimeConv(x,h):

    m = x.size
    n = h.size
    
    x_pad = np.append(x,np.zeros(n))
    h_pad = np.append(h,np.zeros(m))
    
    y = np.zeros(n+m-1)
    for i in np.arange(n+m-1):
        for j in np.arange(m):
            if(i-j+1>0):
                y[i] = y[i]+ (x_pad[j]*h_pad[i-j])
            else:
                break
    
    return y


#If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ?

# The Length of y will be 299 samples


#Convolution using the built in scipy.signal.convolve function 

def CompareConv(x, h): 
    
    time_y1 = time.process_time()
    y1 = myTimeConv(x,h)
    elapsed_time_y1 = time.process_time() - time_y1
    
    
    time_y2 = time.process_time()
    y2 = scipy.signal.convolve(x,h)
    elapsed_time_y2 = time.process_time() - time_y2
   
    diff = y1 - y2
    abs_diff = np.abs(diff)
    m = np.sum(diff) / (y1.size + y2.size)
    mabs = np.sum(abs_diff) / (y1.size + y2.size)
    stdev = np.std(diff)
    
    return m, mabs, stdev, np.array([elapsed_time_y1, elapsed_time_y2])


def main():
    #Generate signals 
    x = np.ones(200)
    h_left = np.arange(0,1,0.04)
    h_right = np.arange(0,1,0.04)[::-1]
    h = np.append(np.append(h_left,1),h_right)
    
    y_time = myTimeConv(x,h)
    
    #Plotting the Convolution

    #Plot x
    plt.figure()
    plt.plot(x)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    #Plot h
    plt.figure()
    plt.plot(h)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    #Plot Convolution of the two signals
    plt.figure()
    plt.title("Convolution of two signals")
    plt.plot(y_time)
    plt.xlabel("n")
    plt.ylabel("y(n)")
    
    plt.savefig('results/01-convolution.png')
    
    
    #Import piano and impulse signals

    (fs_piano,x_piano) = wav.read('piano.wav')
    (fs_impulse_response,x_impulse_response) = wav.read('impulse-response.wav')


    #Convert audio signals into numpy arrays
    
    x_piano = np.array(x_piano, dtype = float)
    x_impulse_response = np.array(x_impulse_response, dtype = float)
    
    #Compute the parameters to compare the two methods of convolution
    #Just first 2000 samples to compare time
    output = CompareConv(x_piano[:2000],x_impulse_response[:2000])
    
    np.savetxt("results/Convolve.txt", output, fmt="%s")


if __name__ == "__main__":
    main()    


