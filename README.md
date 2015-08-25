# Theis
I used Anaconda distribution of python (python 3.4) which comes with a Matlab-like IDE (Spyder) and all necessary libraries (Numpy, Scipy, Matplotlib and others):

http://continuum.io/downloads#py34

The folder analysis scripts contains all the python scripts with comments.

1) 1st run BasicAnalysis.py which loads the results into global variables (it calls OpenLogs.py).
2) Call AverageLeftRight and Interpolate from the GazeAnalysis.py to create necessary variables in the eye traqcking results.
3) Then from GazeAnalysis.py call functions which generate each of the plots from ResultsReport.doc

4) TODO: I will add a Main.py which will call all the steps automatically.

5) TODO: I will save all the data structures (et data and experiment info) into .mat files
