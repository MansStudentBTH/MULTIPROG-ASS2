#include <vector>
#include <algorithm>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
            deviceProp.major, deviceProp.minor);
        
        char msg[256];
        snprintf(msg, sizeof(msg),
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
            (unsigned long long)deviceProp.totalGlobalMem);
        printf("%s", msg);
        printf(
            "\n  GPU Max Clock rate:                            %.0f MHz (%0.2f "
            "GHz)\n",
            deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
            deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n",
            deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
            deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n",
            deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n",
            deviceProp.textureAlignment);
        printf(
            "  Concurrent copy and kernel execution:          %s with %d copy "
            "engine(s)\n",
            (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
            deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
            deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
            deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
            deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
            deviceProp.ECCEnabled ? "Enabled" : "Disabled");

        }
        

}