This is the directory for the 'demoTRISH' sample application.

This demo is based on the original "NVIDIA" CUDA 4.0 "histogram" sample
application.  The original sample uses Podlozhnyuk's histogram method.
Our demo also adds support for a new histogram method called "TRISH".

*TRISH* is a new 256-bin GPU histogram method 
        which is up to 1.5x faster than prior methods for random data.
        and up to 2-4x times faster than prior methods for image data.

TRISH uses Thread level parallelism (Occupancy = 12.5% = 192/1536), 
Instruction Level parallelism (loop unrolling & batching), Vector 
Parallelism (applying arithmetic operations on byte pairs instead of 
4 individual) bytes to speed up overall performance.

More Importantly, TRISH is deterministic and lock-free. IE it doesn't 
use atomics.  This means that you get similar performance regardless 
of the underlying data distribution.  Podlozhnyuk's original histogram 
method's performance will vary depending on how many thread collisions 
the  underlying data causes when binning & counting.


Here are some quick notes.


1.)  The code was written and tested using the following environment
     OS:  Windows 7, SP1
     CUDA:  4.0
     IDE: Microsoft Visual Studio 2008
          Microsoft Visual Studio 2010
     Code Generation:  WIN32, WIN64


2.)  Furthermore, the code was written, developed, and tested
     under the assumption that it is being compiled as 
     part of the demo projects in the NVidia Computing SDK 4.0.  
     IE that we created and stored this directory parallel to all 
     the other sample projects in the CUDA SDK.  
     The Computing 4.0 SDK was found in the following directory 
     on my machine.

c:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.0\C\SRC

     So, this modified "histogram" application should be stored at

...\...\NVIDIA GPU Computing SDK 4.0\c\SRC\demoTRISH

     To build properly...


3.)  List of files.  Here is a list of files that I use to actually
     build my "histogram" demo application.

   // Include files
   histogram_common.h
   
   // Source Files
   histogram256.cu
   histogram64.cu
   histogram_gold.cpp
   histogramTRISH.cu
   main.cpp

The most important file is histogramTRISH.cu that contains the code
that actually implements our *TRISH* method
-- CPU host wrappers
-- Count GPU Kernel
-- RowSum GPU kernel


4.) To get your application to run properly, You may need to 
    include some CUDA DLL's in the same directory as your application 
    or in your windows system directory

    c:\windows\system32 
or 
    c:\windows\SysWOW64

    This might include the following DLL's
       // For 32-bit apps
       cutil32.dll		
       cudart32_40_17.dll

       // For 64-bit apps	
       cutil64.dll      
       cudart64_40_17.dll

    The cutil*.dll files are part of the CUDA 4.0 GPU SDK
    The cudart*.dll files are of the CUDA 4.0 Toolkit
 

6.) Limitations:

** Lack of Generality:
   Currently the TRISH method only supports 8-bit unsigned integer 
   (IE byte) data and assumes 256 bins in the histogram, 
   IE, The GPU method is equivalent to the following CPU algorithm 
   in pseudocode.

   Input:  V = array of 'n' 8-bit bytes to bin & count into histogram
   Output: bins = array of m = 256 bin-counts (IE the final histogram)
   Psuedo-code:

   int bins[256];      
   foreach idx in [0..255] 
      bins[idx] = 0;       // zero counts
   end idx
   foreach idx in [0..n-1] 
      bins[V[idx]]++;      // count bytes
   end idx


** CUDA Enviroment.
    CUDA 3.2 - Should work OK, but I have not tested it.
    CUDA 4.0 - Generates good kernel code (36 registers for Count Kernel)
    CUDA 4.1 - Not so Good, 4.1 Causes the TRISH kernels to blow up 
               Both kernels end up using 63 registers + a large # of 
               register spills.  This negatively impacts performance, 
               for about a 20% loss, caused by dealing with the extra 
               overhead of "Local" memory load/stores for dealing with 
               the register spills.

** Alignment: Assumes input data is actually 32-bit elements (not true byte data)
   -- IE Input data aligned to 32-bit boundaries and nElems is measured in
      32-bit elements (4 bytes) not in actually bytes

** Work Per thread (K-value) in range [1..63]
   We have hardcoded this value to 31 in this example.
   Testing reveals that K=31 and K=63 are the best overall choices
   K=63 achieves maximum throughput for very large values of 'n' 
   (41+ GB/s on GTX 580) However, K=31 is better for smaller to medium 
   values of 'n' as well as achieving good performance on large values 
   of 'n' (40+ GB/s on GTX 580)

   If you want to pick a different 'k' value, change the following line of code

const uint K1_Length    = 31u;		//  31 = Work Per thread (loop unrolling)


** BlockSize: Maximum Threads per Block is 64 (for good performance)
   Shared Memory Usage = 16 KB (64 threads * 64 lanes * 4 bytes) 
   On Fermi cards, this implies 3 concurrent blocks per SM (48KB/16KB)

** GridSize: Best performance is achieved when you specify the Grid Size as 
     #Blocks = #SMs * concurrent blocks per SM
     48 = 16 SMs * 3 concurrent blocks per SM (on GTX 580)
     45 = 15 SMs * 3 concurrent blocks per SM (on GTX 480)
     42 = 14 SMs * 3 concurrent blocks per SM (on Telsa M2050)
     12 =  4 SMs * 3 concurrent blocks per SM (on GTX 560M)

So, don't forget to change the hard coded NUM_GPU_SMs #define in histogramTRISH.cu 
to the correct value for your GPU display card <see below>.  If you set this value incorrectly, you will probably see a significant slowdown in performance.


/*-----------------
  Local Defines
-----------------*/

// GTX 560M
//#define NUM_GPU_SMs (4u)

// TESLA 2050 (2070)
//#define NUM_GPU_SMs (14u)

// GTX 480
//#define NUM_GPU_SMs (15u)

// GTX 580
#define NUM_GPU_SMs (16u)
 