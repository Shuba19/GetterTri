#include "ChronoCuda.h"
chrono_cuda::chrono_cuda(std::string mode)
{
        this->mode = mode;
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);  
        this->stream = 0;     
}

chrono_cuda::chrono_cuda(std::string mode, cudaStream_t stream)
{
        this->mode = mode;
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);  
        this->stream = stream;     
}

void chrono_cuda::cc_start()
{
    cudaEventRecord(this->start, this->stream);
}

void chrono_cuda::cc_stop()
{
    //sync con lo stream
    cudaEventRecord(this->stop, this->stream);
    cudaEventSynchronize(this->stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, this->start, this->stop);
    std::cout << this->mode << " time: " << milliseconds << " ms" << std::endl;
}



chrono_cuda::~chrono_cuda()
{
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->stop);
}


