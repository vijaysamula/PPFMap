# PPFMap

PPFMap is a parallel implementation of the Point Pair Feature matching 
algorithm from [Drost, B.](http://far.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf). The parallel
implementation is as described in the [SLAM++](http://homes.cs.washington.edu/~newcombe/papers/Salas-Moreno_etal_cvpr2013.pdf) project.

### Requirements

To build this project you will need:

+ Point Cloud Library (PCL 1.8.1) : built with CUDA support.
+ Eigen library 3.0
+ CUDA >10
+ cmake 3

### Compile and run

Before compiling the project, check first the cuda capability of your device. 
You can set the specific capability on the `CMakeLists.txt` file, in the 
**CUDA_NVCC_FLAGS**.

```bash
    mkdir build
    cd build
    cmake ..
    make
    make install
    ./demo
```


![Demo preview](https://raw.githubusercontent.com/alfonsoros88/PPFMap/master/doc/images/demo.png)

### Documentation

Doxygen is used for the documentation. To generate it, simply execute the 
following in the `build` directory:

```bash
    make doc
```

