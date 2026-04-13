#ifndef CHRONO_CUDA_H
#define CHRONO_CUDA_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>


class chrono_cuda
{
    cudaEvent_t start, stop;
    cudaStream_t stream;
    std::string mode;

public:
    chrono_cuda(std::string mode);
    chrono_cuda(std::string mode, cudaStream_t stream);
    void cc_start();
    void cc_stop();
    ~chrono_cuda();
};


#endif