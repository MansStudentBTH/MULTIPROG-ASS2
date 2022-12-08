
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <string>


#define TWO_POW_NINTEEN 524288
#define SIZE 128*1024
#define THREADS 256

// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort

// static __device__ inline uint32_t mysmid(){
//     uint32_t smid;
//     asm volatile("mov.u32 %0 , %%smid;" : "=r"(smid));
//     return smid;
// }

// static __device__ inline uint32_t mywarpid(){
//     uint32_t warpid;
//     asm volatile("mov.u32 %0 , %%warpid;" : "=r"(warpid));
//     return warpid;
// }

// static __device__ inline uint32_t __mylaneid(){
//     uint32_t laneid;
//     asm volatile("mov.u32 %0 , %%laneid;" : "=r"(laneid));
//     return laneid;
// }



__global__ void oddEvenGPU(int *arr, int arrSize)
{
    int i = (blockIdx.x*blockDim.x + threadIdx.x) * 2;

	if (i < arrSize) {
		for (int j = 0; j < arrSize/2; j++) {
			for (int k = i; k < arrSize; k += blockDim.x) {
				if (k + 1 < arrSize) {
					if (arr[k] > arr[k + 1]) {
						int temp = arr[k];
						arr[k] = arr[k + 1];
						arr[k + 1] = temp;
					}
				}
			}
			__syncthreads();
			i++;
			for (int k = i; k < arrSize; k += blockDim.x) {
				if (k + 1 < arrSize) {
					if (arr[k] > arr[k + 1]) {
						int temp = arr[k];
						arr[k] = arr[k + 1];
						arr[k + 1] = temp;
					}
				}
			}
			__syncthreads();
			i--;
		}
	}
}

__global__
void oddeven_gpu(int *numbers, unsigned size) {
    unsigned r_size = size;
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int a = 0, b = 0;
    for (int j = 1; j <= r_size; j++)
    {
        for (int i = tid*2+(j%2); i < r_size; i+=(blockDim.x*2)) {
            a = numbers[i];
            b = numbers[i+1];
            if (a>b) {
                numbers[i+1] = a;
                numbers[i] = b;
            }
        }
        __syncthreads();
    }
}




__host__
void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    for (int i = 1; i <= s; i++) {
        for (int j = i % 2; j < s; j = j + 2) {
            if (numbers[j] > numbers[j + 1]) {
                std::swap(numbers[j], numbers[j + 1]);
            }
        }
    }
}

__host__
void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

__host__
int main()
{
    // Initialize a vector with integers of value 0
    std::vector<int> numbers(SIZE);
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    int i = 0;
    std::generate(numbers.begin(), numbers.end(), rand);

    // for(int i = 0; i < SIZE; i++) {
    //     std::cout << numbers[i] << std::endl;
    // }

    // print_sort_status(numbers);
    auto start = std::chrono::steady_clock::now();
    int* arr;
    int data[SIZE];

    if(cudaMalloc((void **)&arr, SIZE*sizeof(int))) {
        printf("Error\n");
    }
    if (cudaMemcpy(arr, numbers.data(), SIZE*sizeof(int), cudaMemcpyHostToDevice)) {
        printf("Error\n");
    }

    oddeven_gpu<<<1,THREADS>>>(arr,SIZE);

    cudaMemcpy(data, arr, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    numbers = std::vector<int>(std::begin(data), std::end(data));
    
    // for(int i = 0; i < SIZE; i++) {
    //     std::cout << numbers[i] << std::endl;
    // }
    // for(int i = 0; i < SIZE; i++) {
    //     std::cout << std::to_string(data[i]) << std::endl;
    // }
    
    cudaFree(arr);
    print_sort_status(numbers);
    //std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";



    //allocate memory space on the device – cudaMalloc(&d_GlblVarPtr, bytes )
    //transfer data from host to device – cudaMemCpy(d_GlblVarPtr, h_Gl...)
    //kernel call – kernelOne<<<execution configuration>>>( args... );
    //transfer results from device to host – cudaMemCpy(h_GlblVarPtr,...)
    //__syncthreads()
}