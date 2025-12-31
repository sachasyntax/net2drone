##net2drone
a network-driven 2-pool audio engine with independent dynamic controls

author: Sacha (Sacha Syntax)

real-time sound engine that converts live network packets into audio while a second RAW byte file modulates various parameters. TCP/UDP packet fields can probabilistically trigger `next track` in either pool.

##features
1.mono 44.1 kHz output  
2.dual feedback memories (direct + accumulator)  
3.circular raw file streaming with moving pointers  
4.independent parameter timers for control pool  
5.non-linear mapping for smoother transitions  
6.TkinterDnD2 drag&drop interface
7.protocol-based pool switching triggers

##parameters
rate, density, level, acc_decay, acc_feedback, dir_feedback, acc_buffersize, probability, blocksize, max_blocksize

##how it works
1. Scapy captures network packets via sniff  
2. packet bytes are converted to floating-point audio  
3. audio is processed by direct memory and accumulator memory with decay and stochastic masking  
4. output audio is normalized, resampled based on rate, hard-clipped toward −1 / 1, and sent to a non-blocking audio queue  
5. control pool bytes update each parameter at different speeds  
6. TCP/UDP packet fields may probabilistically trigger track changes with cooldown

##installation
requires Python 3.8+  
pip install numpy sounddevice scapy tkinterdnd2  
Linux: sudo apt install python3-tk

##usage
1. sudo python net2drone.py  
2. drop files into each pool, 
3. toggle direct/accumulator, 
4. adjust buffer size, 
5. network traffic will generate audio

##probability trigger tuning
NEXT_PROB_TCP = 0.02  
NEXT_PROB_UDP = 0.03  
COOLDOWN = 1.0  
lower values = rarer pool jumps

##curves control
increase sigmoid(k) → stronger movement toward extremes  
bias(pow < 1) → values closer to 0  
bias(pow > 1) → values closer to 1

