# MLLGsecondorder
This is a second order in time and first order in space FEM BEM coupling for the MLLG system. Further details can be found in my thesis https://publikationen.bibliothek.kit.edu/1000133728

run errorplottime.py or errorplotspace.py for a experimental order of convergence experiment either in time or in space.  
The computations are done in the MLLGfun files. Different versions are implemented optimised for either time stepping or space discretization. 

The software runs with 
Fenics 2019.1.0 
Bempp-cl 0.2.0
