import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt

def optRectifierSwtiching(N=2048, outputVoltageSineHarmonicNums=[1,2,4,6],
                          outputVoltageSinevals=[0,0,0,0],
                          outputVoltagecosinharmonicnums=[1,2,4,6],
                          outputVoltagecosinevals=[0,0,0,0],
                          outputVoltageDCval=0.2, gamma=10,
                          solver='ECOS'):
    """
    This function computes the optimal switching of a single phase rectifier
    which minimizes a weighted sum of the THDs of the input current and the output
    voltage.
    :param N: Time discretizations. Must be much larger than the highest harmonic number in the constraints,
    :param outputVoltageSineHarmonicNums: Sine harmonic numbers of the output voltage to be controlled
    :param outputVoltageSinevals: Desired sine harmonic values of the output voltage.
    :param outputVoltagecosinharmonicnums: Cosine harmonic numbers of the output voltage to be controlled
    :param outputVoltagecosinevals: Desired cosine harmonic values of the output voltage.
    :param outputVoltageDCval: Desired DC of the output voltage.
    :param gamma: The weight for the weighted sum of THDs of the input current and the output voltage.
    :param solver: One of the CVX solver. Default is set to ECOS.
    :return: The input current (which also indicates the optimal switching states) and the output voltage.
    """
    time_labels = np.linspace(0,20,2048)

    Fs = np.zeros([len(outputVoltageSineHarmonicNums), N])
    Fc = np.zeros([len(outputVoltagecosinharmonicnums), N])

    for k in range(len(outputVoltageSineHarmonicNums)):
        Fs[k,:] = np.sin(2*np.pi*np.linspace(0,N-1,N)/N*outputVoltageSineHarmonicNums[k])
    for k in range(len(outputVoltagecosinharmonicnums)):
        Fc[k,:] = np.cos(2*np.pi*np.linspace(0,N-1,N)/N*outputVoltagecosinharmonicnums[k])


    Z = cvx.Variable([N,3])
    s = np.array([[-1],[0],[1]])
    prob = cvx.Problem(cvx.Minimize( gamma*np.ones(N)@Z@(s**2)/N + ((np.sin(2*np.pi*np.linspace(0,N-1,N)/N))**2)@(Z*(s**2))/N),
                       [Fc@(np.diag(Fs[0,:])*(Z*s)).flatten() == outputVoltagecosinevals,
                        Fs@(np.diag(Fs[0,:])*(Z*s)).flatten() == outputVoltageSinevals,
                        2*(np.sin(2*np.pi*np.linspace(0,N-1,N)/N))@(Z*s)/N == outputVoltageDCval,
                        np.ones([1,N])*(Z*s/N) == 0,
                        Z >= 0,
                        Z*np.ones([3,1]) == 1])
    prob.solve(solver=solver)

    if(prob.status=='infeasible'):
        print('A solution does not exist with the given constraints!')
        return -1, -1
    elif(prob.status=='optimal_inaccurate'):
        print('The solution is numerically inaccurate. Try using another solver!')

    plt.figure()
    plt.plot(time_labels, np.matmul((Z.value),s),linewidth=3)
    plt.title('Plot of the Switching Scheme/Normalized Current')

    plt.figure()
    plt.plot(time_labels, np.matmul(np.diag(Fs[0,:]),(np.matmul((Z.value),s))),linewidth=3)
    plt.title('Plot of the Output Voltage')


    t = np.matmul(np.diag(Fs[0,:]),(np.matmul((Z.value),s)))
    plt.figure()
    plt.plot(np.abs(np.matmul(sp.linalg.dft(N),t))[0:int(N/2+1)])
    plt.title('Discrete Fourier Transform of the Output Voltage')

    return np.matmul((Z.value),s), np.matmul(np.diag(Fs[0,:]),(np.matmul((Z.value),s)))