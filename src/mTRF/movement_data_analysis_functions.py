### Importing packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import scipy
from scipy.signal import hilbert, chirp
import math
import glob, os
from scipy import interpolate
from os.path import join
from os import listdir
from scipy import signal
from scipy.signal.windows import dpss
from sklearn.metrics.pairwise import cosine_similarity

### Pre-processing

# ~ split the data for two participants and interpolate missing samples)
def split_interpolate(data):
    data_p1 = data.loc[data['Sensor'] == 1]
    data_p2 = data.loc[data['Sensor'] == 2]

    recsamples = list(data_p1["Sample"]) ### recorded samples
    allsamples = list(range(data_p1.iloc[0,1], (data_p1.iloc[-1,1]+1))) ### needed samples

    polhemus_fix_p1 = {}
    polhemus_fix_p2 = {}

    # P1
    # ~ interpolate marker and timestamp columns first
    polhemus_fix_p1["Sample"] = allsamples
    polhemus_fix_p1["Timestamp"] = [round(x) for x in 
                                    interpolate.interp1d(recsamples, data_p1['Timestamp'])(allsamples)]
    polhemus_fix_p1["Marker"] = [round(x) for x in 
                                 interpolate.interp1d(recsamples, data_p1['Marker'])(allsamples)]
    # ~ interpolate movement data
    polhemus_fix_p1["Position X"] = interpolate.PchipInterpolator(recsamples, data_p1['Position X'])(allsamples)
    polhemus_fix_p1["Position Y"] = interpolate.PchipInterpolator(recsamples, data_p1['Position Y'])(allsamples)
    polhemus_fix_p1["Position Z"] = interpolate.PchipInterpolator(recsamples, data_p1['Position Z'])(allsamples)

    # P2
    # ~ interpolate marker and timestamp columns first
    polhemus_fix_p2["Sample"] = allsamples
    polhemus_fix_p2["Timestamp"] = [round(x) for x in 
                                    interpolate.interp1d(recsamples, data_p2['Timestamp'])(allsamples)]
    polhemus_fix_p2["Marker"] = [round(x) for x in 
                                 interpolate.interp1d(recsamples, data_p2['Marker'])(allsamples)]
    # ~ interpolate movement data
    polhemus_fix_p2["Position X"] = interpolate.PchipInterpolator(recsamples, data_p2['Position X'])(allsamples)
    polhemus_fix_p2["Position Y"] = interpolate.PchipInterpolator(recsamples, data_p2['Position Y'])(allsamples)
    polhemus_fix_p2["Position Z"] = interpolate.PchipInterpolator(recsamples, data_p2['Position Z'])(allsamples)

    return(polhemus_fix_p1,polhemus_fix_p2)

# ~ separate trils into conditions
def return_by_condition(p1,n_tr):
    da = pd.DataFrame(p1)['Marker']
    trial_onsets = [i+1 for i in range(0,len(pd.DataFrame(p1)['Marker'])-1) if (da[i]==0) and (da[i+1] == 1)]
    trial_onsets = np.unique([trial_onsets[i-1] for i in range(1,len(trial_onsets)) 
                              if trial_onsets[i]-trial_onsets[i-1] < 5000 and trial_onsets[i]-trial_onsets[i-1] > 4700] + [trial_onsets[i] for i in range(1,len(trial_onsets)) 
                                                                                                                           if trial_onsets[i]-trial_onsets[i-1] < 5000 and trial_onsets[i]-trial_onsets[i-1] > 4700])
    trial_onsets = trial_onsets[-n_tr*3:]
    c1 = [pd.DataFrame(p1).iloc[i:i+20*240,:] for i in trial_onsets[::3]]
    c2 = [pd.DataFrame(p1).iloc[i:i+20*240,:] for i in trial_onsets[1::3]]
    c3 = [pd.DataFrame(p1).iloc[i:i+20*240,:] for i in trial_onsets[2::3]]
    return(c1,c2,c3)


# ~ extract phase by transforming Z & X dimensions to polar coordinates
def zscore_filter_phase(trials,fc):
    n = len(trials)
    x_zscore = [np.array(scipy.stats.zscore(trials[i]['Position X'])) for i in range(0,n)] # vertical (i.e., y)
    z_zscore = [np.array(scipy.stats.zscore(trials[i]['Position Z'])) for i in range(0,n)] # horizontal (i.e., x)

    x_filt = [mne.filter.filter_data(data = x_zscore[i],sfreq = 240,l_freq = None,h_freq = fc,method='fir',verbose=False) for i in range(0,n)]
    z_filt = [mne.filter.filter_data(data = z_zscore[i],sfreq = 240,l_freq = None,h_freq = fc,method='fir',verbose=False) for i in range(0,n)]
    
    # x is vertical and z is horizontal
    def phase_fun(x,z):
        phase = []
        for i in range(0,len(z)):
            phase.append(math.atan2(x[i],z[i]))
        return(phase)
    
    phase_data = []
    for i in range(0,n):
        phase_data.append(phase_fun(x_filt[i],z_filt[i]))
    
    return(np.array(phase_data))

# ~ extract phase from 1D arrays (X)
def zscore_filter_phase_1d(trials,fc):
    n = len(trials)
    x_zscore = [np.array(scipy.stats.zscore(trials[i]['Position X'])) for i in range(0,n)]
    x_filt = [mne.filter.filter_data(data = x_zscore[i],sfreq = 240,l_freq = None,h_freq = fc,method='fir',verbose=False) for i in range(0,n)]

    analytic_signal = hilbert(x_filt)
    phase = np.angle(analytic_signal)
    return(phase)

### A simpler function – for the sake of visualization
def zscore_filter(trials,l_freq,h_freq):
    n = len(trials)
    x_zscore = [np.array(scipy.stats.zscore(trials[i]['Position X'])) for i in range(0,n)]
    z_zscore = [np.array(scipy.stats.zscore(trials[i]['Position Z'])) for i in range(0,n)]

    x_filt = [mne.filter.filter_data(data = x_zscore[i],sfreq = 240,l_freq = l_freq,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    z_filt = [mne.filter.filter_data(data = z_zscore[i],sfreq = 240,l_freq = l_freq,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    
    return(x_filt,z_filt)

def zscore_filter_1d(trials,h_freq):
    n = len(trials)
    x_zscore = [np.array(scipy.stats.zscore(trials[i]['Position X'])) for i in range(0,n)]

    x_filt = [mne.filter.filter_data(data = x_zscore[i],sfreq = 240,l_freq = None,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    return(x_filt)

### A simpler function – for the sake of visualization
def zscore_filter_3d(trials,h_freq):
    n = len(trials)
    x_zscore = [np.array(scipy.stats.zscore(trials[i]['Position X'])) for i in range(0,n)]
    z_zscore = [np.array(scipy.stats.zscore(trials[i]['Position Z'])) for i in range(0,n)]
    y_zscore = [np.array(scipy.stats.zscore(trials[i]['Position Y'])) for i in range(0,n)]

    x_filt = [mne.filter.filter_data(data = x_zscore[i],sfreq = 240,l_freq = None,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    z_filt = [mne.filter.filter_data(data = z_zscore[i],sfreq = 240,l_freq = None,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    y_filt = [mne.filter.filter_data(data = y_zscore[i],sfreq = 240,l_freq = None,h_freq = h_freq,method='fir',verbose=False) for i in range(0,n)]
    
    return(x_filt,z_filt,y_filt)


### CCOR functions
# ~ Hard-coded from the original formula
def ccor_custom(x,y):
    def circular_mean(radians):
    # Calculate the sum of sin and cos values
        sin_sum = sum([math.sin(rad) for rad in radians])
        cos_sum = sum([math.cos(rad) for rad in radians])

        # Calculate the circular mean using arctan2
        mean_rad = math.atan2(sin_sum, cos_sum)
        return mean_rad
    x_mean = circular_mean(x)
    y_mean = circular_mean(y)

    numerator = []
    for i in range(0,len(x)):
        numerator.append((np.sin(x[i]-x_mean))*(np.sin(y[i]-y_mean)))
    numerator = sum(numerator)

    denominator_x = []
    denominator_y = []
    for i in range(0,len(x)):
        denominator_x.append((np.sin(x[i]-x_mean))**2)
        denominator_y.append((np.sin(y[i]-y_mean))**2)
    denominator_x = np.sqrt(sum(denominator_x))
    denominator_y = np.sqrt(sum(denominator_y))
    denominator = denominator_x * denominator_y

    ccor = numerator/denominator
    return(ccor)

# ~ Assuming the direction is meaningless (the data does not follow von Mises distr, so the mean direction is meaningless)
# Remark 2 in https://jammalam.faculty.pstat.ucsb.edu/html/Some%20Publications/1988_Correlation%20Coefficient%20for%20Angular%20Variables_STDAII.pdf
# Means are CHOSEN because they are arbitrary, since the data does not follow von Mises

def circ_mean(alpha):
    return np.angle(np.sum(np.exp(1j * alpha)))

# ~ relative phase function
def relative_phase(target,source,type):
    delta_phi = target - source
    # ~ if the input is an array
    if type=='array':
        for i in range(0,len(delta_phi)):
            if delta_phi[i] > np.pi:
                delta_phi[i] -= 2*np.pi
            elif delta_phi[i] < -np.pi:
                delta_phi[i] += 2*np.pi
    # ~ if the input is a single number
    elif type=='number':
            if delta_phi > np.pi:
                delta_phi -= 2*np.pi
            elif delta_phi < -np.pi:
                delta_phi += 2*np.pi
    return delta_phi

def new_rel_phase(phase1, phase2): # ~ a vectorized version of the function below
    return np.angle(np.exp(1j * (phase1 - phase2)))

#def new_rel_phase(phase1,phase2):
#    relative_phase = phase1 - phase2
#    relative_phase = [np.angle(np.exp(1j * i)) for i in relative_phase]
#    return(relative_phase)

def ccorr_uniform(alpha1, alpha2):
    a_minus_b = alpha1-alpha2
    a_plus_b = alpha1+alpha2

    amb_mean = circ_mean(a_minus_b)
    apb_mean = circ_mean(a_plus_b)

    # solving m-n=mean_dir(a-b) and m+n=mean_dir(a+b):
    # m-n-m-n = mean_dir(a-b)-mean_dir(a+b)
    # -2n = mean_dir(a-b)-mean_dir(a+b)
    # n = (mean_dir(a-b)-mean_dir(a+b)) / (-2)
    # m = mean_dir(a-b)+n

    # if I understand the logic correctly, all summation operations are treated as relative phase since the data is angular

    nu = (amb_mean-apb_mean)/(-2)
    mu = (amb_mean+nu)

    x_sin = np.sin(alpha1 - mu)
    y_sin = np.sin(alpha2 - nu)
    r_minus = np.abs(np.sum(np.exp((alpha1-alpha2) * 1j)))
    r_plus  = np.abs(np.sum(np.exp((alpha1+alpha2) * 1j)))
    num = (r_minus - r_plus)
    den = 2*np.sqrt(np.sum(x_sin**2) * np.sum(y_sin**2))
    rho = num / den
    return(rho)

# ~ Calculating PLV and PLI from the formula (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3674231/)

def plv_custom(x,y):
    N = len(x)
    PLV = abs((1/N)*sum(np.exp(1j*relative_phase(x,y,'array'))))
    return PLV

def pli_custom(x,y):
    N = len(x)
    PLI = abs((1/N)*sum(np.sign(relative_phase(x,y,'array'))))
    return PLI


### With moving window (window size and interval [=resolution] in seconds)
def ccors_window(x,y,kind,window_size,resolution):
    ws = int(window_size*240)
    r = int(resolution*240)
    l = int(len(x[0])/r-(ws/r-1))
    ccors_window = []
    for k in range(0,20):
        for i in range(0,l):
            if kind == "old_ccor":
                ccors_window.append(ccor_custom(x[k][r*i : r*i + ws],y[k][r*i : r*i + ws]))
            elif kind == "ccorr_uniform":
                ccors_window.append(ccorr_uniform(x[k][r*i : r*i + ws],y[k][r*i : r*i + ws]))
            elif kind == "plv":
                ccors_window.append(plv_custom(x[k][r*i : r*i + ws],y[k][r*i : r*i + ws]))
            elif kind == "pli":
                ccors_window.append(pli_custom(x[k][r*i : r*i + ws],y[k][r*i : r*i + ws]))
    ccors_window = np.array(ccors_window)
    ccors_window = np.reshape(ccors_window, (20,l))
    return(ccors_window)

### By splitting the data into non-overlapping segments
def ccors_segm(x,y):
    ccors_segm = []
    for k in range(0,20):
        for i in range(0,20,1):
            ccors_segm.append(ccorr_uniform(x[k][250*i : 250*i + 250*1],y[k][250*i : 250*i + 250*1]))
    ccors_segm = np.array(ccors_segm)
    ccors_segm = np.reshape(ccors_segm, (20,20))
    return(ccors_segm)


### Function for detecting the direction of rotation
def direction_of_rotation(phase_array):
    result = [0]*(len(phase_array))
    res = 5
    for i in range(res,(len(phase_array))):
        ### counterclockwise
        if (phase_array[i]>phase_array[i-res]) and (abs(phase_array[i]-phase_array[i-res])<5):
            result[i] = 1
        ### accounting for the transition from pi to zero for a new cycle
        elif (abs(phase_array[i]-phase_array[i-res])>5) and (phase_array[i]<phase_array[i-res]):
            result[i] = 1
        ### clowckwise
        elif (phase_array[i]<phase_array[i-res]) and (abs(phase_array[i]-phase_array[i-res])<5):
            result[i] = -1
        ### accounting for the transition from zero to pi for a new cycle
        elif (abs(phase_array[i]-phase_array[i-res])>5) and (phase_array[i]>phase_array[i-res]):
            result[i] = -1
    ### removing values appearing over an extremely short duration (i.e., jitter)
    change_points = [0]*len(result)
    for i in range(1,len(result)):
        if result[i] != result[i-1]:
            change_points[i] = 1
    change_indices = [i for i in range(0,len(change_points)) if change_points[i]==1]
    small_groups = []
    for k in range(1,len(change_indices)):
        if (change_indices[k]-change_indices[k-1]) < 10:
            small_groups.append((change_indices[k-1],change_indices[k]))
    for k in range(0,len(small_groups)):
        for i in range((small_groups[k][0]),(small_groups[k][1]+1)):
                result[i] = result[i-1]
    final_result = [0]*len(result)
    for i in range(1,len(final_result)):
        if result[i] > result[i-1]:
            final_result[i] = 1
        elif result[i] < result[i-1]:
            final_result[i] = -1
    return final_result

def clear_zeros(ar):
    ar2 = ar.copy()
    for i in range(0,len(ar2)):
        if ar2[i] == 0:
            ar2[i] = None
    return ar2


def plot_all_trials(p1,p2,ws,res,col1,col2):
    plt.figure(figsize=(16, 100), dpi=150)
    
    old_c = ccors_window(p1,p2,'old_ccor',ws,res)
    new_c = ccors_window(p1,p2,'ccorr_uniform',ws,res)
    plv = ccors_window(p1,p2,'plv',ws,res)
    pli = ccors_window(p1,p2,'pli',ws,res)

    for k in range(0,20):
        plt.subplot(60,1,k*3+1)
        plt.plot(p1[k],label="Participant 1", color = col1)
        plt.plot(p2[k], label="Participant 2", color = col2)
        plt.title(f'Trial {k}')

        plt.subplot(60,1,k*3+2)
        xxx_p1 = direction_of_rotation(p1[k])
        xxx_p2 = direction_of_rotation(p2[k])
        plotting_p1 = clear_zeros(xxx_p1)
        plotting_p2 = clear_zeros(xxx_p2)
    
        plt.plot([(i,i) for i in range(0,len(xxx_p1))],[(0,i) for i in xxx_p1],linestyle='--',color=col1,alpha=0.5)
        plt.plot(plotting_p1,marker='o',color=col1,alpha=0.5,markersize=15)

        plt.plot([(i,i) for i in range(0,len(xxx_p2))],[(0,i) for i in xxx_p2],linestyle='--',color=col2,alpha=0.5)
        plt.plot(plotting_p2,marker='*',color=col2,alpha=1,markersize=10)
        plt.title('Direction of movement')

        plt.subplot(60,1,k*3+3)
    
        plt.plot(old_c[k],'-o',color='black',label='CCor (old)')
        plt.plot(new_c[k],'-o',color='red',label='CCor (uniform)')
        plt.plot(plv[k],'-o',color='green',label='PLV')
        plt.plot(pli[k],'-o',color='magenta',label='PLI')
        plt.axhline(y=0,linestyle='--',color='black',alpha=0.2)
        plt.xticks(range(0,len(old_c[k])))
        plt.ylim(-1,1)
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


### Binning function for relative phase

def bin_rel_phase(mean_rel_phase,n_bins):
    step = (math.pi-(-math.pi))/n_bins
    binning = [(-math.pi+step*i) for i in range(0,(n_bins+1))]
    y_ax, x_ax_prep = np.histogram(mean_rel_phase, bins = binning)
    x_ax = [np.mean([x_ax_prep[i],x_ax_prep[i+1]]) for i in range(0,len(x_ax_prep)-1)]
    return(x_ax,y_ax)


### A utility function for flatteing arrays

def flatten(xss):
    return [x for xs in xss for x in xs]

### A utility function for averaging across trials

def averages_across_trials(data):
    c0 = []
    c1 = []
    c2 = []
    for pair in range(0,len(data)):
        t_list0 = []
        t_list1 = []
        t_list2 = []
        for t in range(0,20):
            t_list0.append(abs(data[pair][0][t]))
            t_list1.append(abs(data[pair][1][t]))
            t_list2.append(abs(data[pair][2][t]))
        c0.append(np.median(t_list0))
        c1.append(np.median(t_list1))
        c2.append(np.median(t_list2))
    return(c0,c1,c2)

### A utility function for calculating windowed ccor

def get_windowed_results(data,ws,step,thr,measure):
    N = len(data)
    cond0_w = []
    cond1_w = []
    cond2_w = []
    for pair in range(0,N):
        cond0_w.append(len([i for i in abs(ccors_window(data[pair][0][0],data[pair][1][0],measure,ws,step).flatten()) if i >= thr]))
        cond1_w.append(len([i for i in abs(ccors_window(data[pair][0][1],data[pair][1][1],measure,ws,step).flatten()) if i >= thr]))
        cond2_w.append(len([i for i in abs(ccors_window(data[pair][0][2],data[pair][1][2],measure,ws,step).flatten()) if i >= thr]))
    windowed_results = [cond0_w,cond1_w,cond2_w]
    return(windowed_results)

### PSD

def multitaper_spectrum(data, fs, NW=4, num_tapers=None, adaptive=True, return_weights=False):
    """
    Calculate the multitaper power spectrum of a time series using DPSS windows.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    fs : float
        Sampling frequency
    NW : float, optional (default=4)
        Time-bandwidth product. Determines the amount of spectral smoothing.
    num_tapers : int, optional
        Number of DPSS tapers to use. If None, will use 2*NW-1
    adaptive : bool, optional (default=True)
        Whether to use adaptive weighting
    return_weights : bool, optional (default=False)
        Whether to return the adaptive weights used
        
    Returns:
    --------
    frequencies : ndarray
        Array of frequencies
    psd : ndarray
        Power spectral density estimate
    weights : ndarray, optional
        Adaptive weights used (if return_weights=True)
    """
    # Convert input to numpy array
    data = np.array(data)
    N = len(data)
    
    # If num_tapers not specified, use optimal number
    if num_tapers is None:
        num_tapers = int(2 * NW - 1)
    
    # Generate DPSS windows
    tapers, eigenvalues = dpss(N, NW, num_tapers, return_ratios=True)
    
    # Calculate FFT of tapered data
    tapered_data = data[np.newaxis, :] * tapers
    fft_data = np.fft.fft(tapered_data, axis=1)
    
    # Calculate frequencies
    frequencies = np.fft.fftfreq(N, d=1/fs)
    
    if adaptive:
        # Adaptive weights
        weights = _adaptive_weights(fft_data, eigenvalues)
        psd = np.abs(fft_data)**2
        psd_weighted = np.sum(weights * psd, axis=0) / np.sum(weights, axis=0)
    else:
        # Simple averaging
        weights = np.ones((num_tapers, N)) / num_tapers
        psd = np.mean(np.abs(fft_data)**2, axis=0)
        psd_weighted = psd
    
    # Organize frequencies and PSD to return positive frequencies only
    mask = frequencies >= 0
    frequencies = frequencies[mask]
    psd_weighted = psd_weighted[mask]
    
    if return_weights:
        return frequencies, psd_weighted, weights[:, mask]
    return frequencies, psd_weighted

def _adaptive_weights(fft_data, eigenvalues):
    """
    Calculate adaptive weights for multitaper spectral estimation.
    
    Parameters:
    -----------
    fft_data : ndarray
        FFT of tapered data
    eigenvalues : ndarray
        Eigenvalues of DPSS tapers
        
    Returns:
    --------
    weights : ndarray
        Adaptive weights for each taper at each frequency
    """
    power = np.abs(fft_data)**2
    N = power.shape[1]
    
    # Initial spectrum estimate
    psd_init = np.mean(power, axis=0)
    
    # Adaptive weights
    weights = np.empty_like(power)
    for i in range(N):
        weights[:, i] = eigenvalues / (eigenvalues + (1 - eigenvalues) * power[:, i] / psd_init[i])
    
    return weights


# Function to compute cosine similarity
def compute_cos_sim(vec1, vec2):
    vec1 = vec1.reshape(1, -1)  # Reshape for sklearn
    vec2 = vec2.reshape(1, -1)  # Reshape for sklearn
    return cosine_similarity(vec1, vec2)[0, 0]


### Circular tests and supporting functions

def rotate_circular_distribution(angles, rotation_angle):
    """
    Rotate a circular distribution of angles while keeping them within [-π, π].
    
    Parameters:
    -----------
    angles : array-like
        Array of angles in radians, assumed to be in range [-π, π]
    rotation_angle : float
        Angle to rotate by, in radians
        
    Returns:
    --------
    rotated_angles : numpy.ndarray
        Rotated angles, wrapped to stay within [-π, π]
    """
    # Convert input to numpy array if it isn't already
    angles = np.asarray(angles)
    
    # Verify input angles are within [-π, π]
    if np.any(angles < -np.pi) or np.any(angles > np.pi):
        raise ValueError("Input angles must be within [-π, π]")
        
    # Add the rotation angle
    rotated = angles + rotation_angle
    
    # Wrap the angles to stay within [-π, π]
    wrapped = np.mod(rotated + np.pi, 2 * np.pi) - np.pi
    
    return wrapped

def calculate_resultant_length(angles):
    """
    Calculate the resultant length R for a set of angles.
    
    Parameters:
    angles : array-like
        Angular measurements in radians
    
    Returns:
    float : The resultant length R
    """
    n = len(angles)
    # Convert to rectangular coordinates
    cos_sum = np.sum(np.cos(angles))
    sin_sum = np.sum(np.sin(angles))
    
    # Calculate mean coordinates
    x_bar = cos_sum / n
    y_bar = sin_sum / n
    
    # Calculate R
    R = n * np.sqrt(x_bar**2 + y_bar**2)
    return R

def watson_williams_test(sample1, sample2):
    """
    Perform Watson-Williams test for comparing two samples of circular data.
    
    Parameters:
    sample1 : array-like
        First sample of angles in radians
    sample2 : array-like
        Second sample of angles in radians
    
    Returns:
    float : F-statistic for the Watson-Williams test
    
    Note:
    - The test assumes von Mises distribution
    - Assumes similar concentration parameters between groups

    Watson, G. S., & Williams, E. J. (1956). On the construction of significance tests on the circle and the sphere. Biometrika, 43(3/4), 344-352. FORMULA 22.

    """
    # Convert inputs to numpy arrays
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    # Calculate resultant lengths
    R1 = calculate_resultant_length(sample1)
    R2 = calculate_resultant_length(sample2)
    
    # Combine samples and calculate total resultant length
    combined = np.concatenate([sample1, sample2])
    R = calculate_resultant_length(combined)
    
    # Total number of observations
    N = len(sample1) + len(sample2)
    
    # Calculate F statistic
    F = ((N - 2) * (R1 + R2 - R)) / (1 * (N - (R1 + R2)))
    
    return F

# Bootstrapping function
def bootstrap_watson_williams(group1, group2, n_iterations=1000):
    # Observed statistic
    observed_stat = watson_williams_test(group1, group2)
    
    # Combine the two groups for resampling
    combined_data = np.concatenate([group1, group2])
    
    # Initialize a list to store bootstrap statistics
    bootstrap_stats = []
    
    # Perform bootstrapping
    for _ in range(n_iterations):
        # Resample (with replacement) the combined data
        sample1 = np.random.choice(combined_data, size=len(group1), replace=True)
        sample2 = np.random.choice(combined_data, size=len(group2), replace=True)
        
        # Calculate Watson-Williams statistic for the resampled groups
        bootstrap_stats.append(watson_williams_test(sample1, sample2))
    
    # Calculate p-value as the proportion of bootstrap stats greater than the observed stat
    p_value = np.sum(np.array(bootstrap_stats) >= observed_stat) / n_iterations
    return observed_stat, np.array(bootstrap_stats), p_value

def circ_tests(data):
    """
    A wrapper function for a series of statistical tests to check if a circular distribution is different from a uniform.
    If yes, then test if its circular mean is different from zero and from pi.
    """
    ### 1. Test if the distribution is different from the uniform
    import pingouin as pg

    rayleigh_test = pg.circ_rayleigh(data)
    uniformity_test = rayleigh_test[1]<0.01  # ~ if significant, then NOT uniform. We want SIGNIFICANT below 0.01
    if uniformity_test:
        ### 2. Test if the mean is different from 0
        real_mean = circ_mean(data)
        data_shifted_to_zero = rotate_circular_distribution(data, 0-real_mean)
        _, _, p_value_zero = bootstrap_watson_williams(data, data_shifted_to_zero, n_iterations=1000)
        zero_mean_test = p_value_zero<0.05  # ~ if significant, then DIFFERENT. We want SIGNIFICANT below 0.05
        zero_mean_test = not zero_mean_test

        ### 3. Test if the mean is different from pi
        data_shifted_to_pi = rotate_circular_distribution(data, np.pi-real_mean)
        _, _, p_value_pi = bootstrap_watson_williams(data, data_shifted_to_pi, n_iterations=1000)
        pi_mean_test = p_value_pi<0.05  # ~ if significant, then DIFFERENT. We want SIGNIFICANT below 0.05
        pi_mean_test = not pi_mean_test
    else:
        zero_mean_test = None
        pi_mean_test = None
    return({'uniform': not uniformity_test,
            'zero_mean': zero_mean_test,
            'pi_mean':pi_mean_test})

def mic_custom(x,y):
    from minepy import MINE

    mine = MINE()
    mine.compute_score(x, y)
    mic = mine.mic()
    return(mic)


"""
Source of the dPLI formula:
Stam, C. J., & van Straaten, E. C. (2012). Go with the flow: use of a directed phase lag index (dPLI) to characterize patterns of phase relations in a large-scale model of brain dynamics. Neuroimage, 62(3), 1415-1428.
I ended up using the non-abs PLI only because I don't need to logit-transform it

            trial_data = np.array(trial_data)
            heaviside = np.zeros_like(trial_data)
            heaviside[trial_data > 0] = 1
            heaviside[trial_data == 0] = 0.5
            dpli = np.mean(heaviside)

            
"""


### Defining a segmenting function for cutting original EEG epochs

def segment_epochs(full_epochs, condition, segm_length_sec):
    duration = len(full_epochs[condition].get_data()[0][0])/256
    n_segm = math.floor(duration/segm_length_sec)
    epochs_list = []
    x = 0
    k = 0 ### a time shifter to make sure all shorter epochs have the same times
    for i in range(0,n_segm):
        cond_epochs = mne.EpochsArray(data = full_epochs[condition].get_data(),info=full_epochs.info,verbose=False)
        shorter_epochs = cond_epochs.crop(tmin=x,tmax=x+segm_length_sec,include_tmax=False,verbose=False)
        shorter_epochs.shift_time(k, relative=True)
        epochs_list.append(shorter_epochs)
        k = -x-segm_length_sec
        x = x+segm_length_sec
    conc_object = mne.concatenate_epochs(epochs_list, add_offset=True,verbose=False) ### concatenate
    final_object = mne.EpochsArray(data = conc_object.get_data(), info = conc_object.info,verbose=False) ### resent indices
    return(final_object)

### Summing and take a difference of two consecutive values (no overlap) in arrays
def sum_pairs(arr):
    return [arr[i] + (arr[i+1] if i+1 < len(arr) else 0)
            for i in range(0, len(arr), 2)]

def diff_pairs(arr):
    return [arr[i] - (arr[i+1] if i+1 < len(arr) else 0)
            for i in range(0, len(arr), 2)]