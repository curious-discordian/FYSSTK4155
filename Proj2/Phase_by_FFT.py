### ------------------ Phase analysis : --------------------------------- 
## By basic Fourier, we should be able to discern the probability that
## We are dealing with a stable/ordered phase.
##
## Please note that the probabilities are not calibrated here, but
## we can think of this as a pseudo-probability.
##
## It seems good enough for the measure of whether something is stable
## or not.

def FFT_analysis(t,show_demo=False):
    """
    Pass a temperature in the valid data-pool. 
    optionally show_demo [boolean] to show an example. 
    Returns the percentage of samples at that point where the order
    is over 80%. 
    (Seemed like a fair place to start) 
    """
    def FFT_phase(data):
        # Assuming the data is singular here.
        analysis = np.abs(np.fft.fftshift(np.fft.fft2(data.reshape(L,L))))
        # Note the shift-function, which transposes q1 with q3 and q2 with q4.
        # This is a pretty standard way of dealing with FFT2, as the
        # positive frequencies are on the first half, and the negative on the second.
        analysis = normalize(analysis)
        return analysis

    data_read = read_t(t) 
    FFT = np.array([FFT_phase(x.reshape(L,L)) for x in data_read])
    # Now for the hypothesis; it seems like the data is ordered
    # when the max value is over about .8,
    # and unordered if it is not.
    # So let's take the max of each of these, and then 
    maxes = np.array([np.max(x) for x in FFT])
    percentage = np.sum(np.where(maxes>.80,1,0))/float(len(maxes))
    if show_demo:
        # Should ensure that the pick is representative of the
        # genereral cases (i.e. pick the mean max-value) 
        pick = np.argsort(maxes)[len(maxes)//2] #index of median value
        data = data_read[pick]
        plt.figure()
        plt.subplot(211)
        plt.contourf(data.reshape(L,L))
        xx,yy = np.meshgrid(np.linspace(-L/2,L/2,L),np.linspace(-L/2,L/2,L))
        plt.subplot(212)
        plt.contourf(xx,yy,FFT[pick])

        plt.show()
        
    return percentage
