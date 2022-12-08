#include <stdio.h>

const int SIZE = 16;
const int BLOCK_SIZE = SIZE;
const int NUM_BLOCKS = 1;

void printData(int *data, int size);

__global__
void addOne(int *data, int size)
{
    int tid = threadIdx.x;
    if (tid < size)
        data[tid] += 1;
}

int main()
{
    // Allocate memory on the host and initialize some values
    int *data_h = new int[SIZE];
    for (int i = 0; i < SIZE; i++)
        data_h[i] = rand() % 100;

    printData(data_h, SIZE);

    // Allocate memory on the device, then copy the host data to the device
    int *data_d;
    cudaMalloc(&data_d, SIZE * sizeof(int));
    cudaMemcpy(data_d, data_h, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    addOne<<<NUM_BLOCKS, BLOCK_SIZE>>>(data_d, SIZE);

    // Copy the sorted data from the device to the host
    cudaMemcpy(data_h, data_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print some stuff
    printf("Sorted data:\n");
    printData(data_h, SIZE);

    // Free some memory
    cudaFree(data_d);
    delete[] data_h;
    return 0;
}

void printData(int *data, int size)
{
    for (int i = 0; i < size; i++)
        printf("%d ", data[i]);
    printf("\n");
}