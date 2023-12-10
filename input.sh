#!/bin/bash

# exeName

# input x-v-data file name
input="1"

# output base filename
basename="0.1"

# Dimension of the simulation
L0=70
L1=60
L2=60

# density of mpcd particles:
d=10

#number of rings:
n=1
#number of monomer in each ring 
m=1

# shear rate
s=0.0

#md time step:
hmd=0.002

#mpcd time step:
hmpcd=0.01

# output intervall 
Tout=1

# simulate time 
Tsim=100

#start time: 
t=0

#topology 0 for ring 1 for poltycatenane 2 for linked ring , 3 for ... and 4 for one active particle in a fluid.
topology=4

#Activity on or off (0 o1 1):
Activity=1

#Random flag :
random_flag=0

./main.run  $input $basename $L0 $L1 $L2 $d $n $m $s $hmd $hmpcd $Tout $Tsim $t $topology $Activity $random_flag




