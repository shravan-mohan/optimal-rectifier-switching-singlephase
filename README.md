# Optimal Rectifier Switching (Single Phase)
This function computes the optimal switching of a single phase rectifier which minimizes a weighted sum of the THDs of the input current and the output voltage,
 using linear programming. 

# Use
optRectifierSwtiching(N=2048, outputVoltageSineHarmonicNums=[1,2,4,6],
                          outputVoltageSinevals=[0,0,0,0],
                          outputVoltagecosinharmonicnums=[1,2,4,6],
                          outputVoltagecosinevals=[0,0,0,0],
                          outputVoltageDCval=0.2, gamma=10,
                          solver='ECOS')
                          
# Package Requirements
1. Numpy
2. Scipy 
3. CVXPY 
4. Matplotlib
