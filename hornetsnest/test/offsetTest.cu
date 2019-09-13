/**
 * @brief Connected-Component test program
 * @file
 */


#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

#include <Hornet.hpp>
#include <HornetAlg.cuh>
#include <BasicTypes.hpp>

#include <Device/Util/Timer.cuh>


namespace hornets_nest{
    using string_t = char;
    using soffset_t = int;

    using HornetInit  = ::hornet::HornetInit<string_t,EMPTY,EMPTY,soffset_t>;
    // using HornetDynamicGraph = ::hornet::gpu::Hornet<string_t>;
    using HornetStaticGraph = ::hornet::gpu::HornetStatic<string_t>;

}

#define CHECK_ERROR(str) \
    {cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}

#include "offsetKernels.cuh"

template <typename HornetGraph>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // soffset_t wordCount   = 3;
    // soffset_t letterCount = 11; 
    // string_t letters[letterCount]       = {'m','y','f','i','r','s','t','t','e','s','t'};
    // soffset_t wordOffsets[wordCount+1]  = {0,2,7,11};

    // HornetInit hornet_init(wordCount,letterCount,wordOffsets,letters);
    // HornetStaticGraph hgraph(hornet_init);

    int* dCounter = NULL;
    // string_t myChar = 't';
    cudaMalloc((void**)&dCounter, sizeof(int));
    cudaMemset(dCounter,0, sizeof(int));

    // load_balancing::BinarySearch load_balancing(hgraph);

    // forAllEdges(hgraph, hornets_nest::findAndCount {myChar, dCounter },load_balancing);



    // forAll(, hornets_nest::findAndCount {myChar, dCounter },load_balancing);

    cudaEventRecord(start); 
    cudaEventSynchronize(start); 
    if(argc >=2 ){
        FILE *f = fopen(argv[1], "r");
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

        string_t *h_fileInfo = (string_t *)malloc(fsize + 1);
        size_t readSize = fread(h_fileInfo, 1, fsize, f);
        fclose(f);

        h_fileInfo[fsize++] = '\n';    

        string_t* d_fileInfo;
        cudaMalloc((string_t**)&d_fileInfo, sizeof(string_t)*fsize);
        cudaMemcpy(d_fileInfo,h_fileInfo, sizeof(string_t)*fsize, cudaMemcpyHostToDevice);
        cudaMemset(dCounter,0, sizeof(int));

        string_t myChar='\n';
        forAll(fsize, hornets_nest::findAndCountArray {myChar, dCounter, d_fileInfo });

        int hlineCounts = 0;
        cudaMemcpy(&hlineCounts,dCounter,sizeof(int), cudaMemcpyDeviceToHost);
        printf("Line Count : %d \n",hlineCounts);

        soffset_t *rowStarts,*rowStartsSorted;//,*rowOffsets;

        gpu::allocate(rowStartsSorted,hlineCounts+1);
        gpu::allocate(rowStarts,hlineCounts+1);
        // gpu::allocate(rowOffsets,hlineCounts+1);
        cudaMemset(dCounter,0, sizeof(int));
        cudaMemset(rowStarts,0, sizeof(int));

        forAll(fsize, hornets_nest::findAndStoreLineEnds {'\n', dCounter, d_fileInfo,rowStarts+1});

        cudaMemcpy(&hlineCounts,dCounter,sizeof(int), cudaMemcpyDeviceToHost);
        printf("Line Count : %d \n",hlineCounts);


        // Sorting the starting positions
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, rowStarts, rowStartsSorted, hlineCounts+1);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, rowStarts, rowStartsSorted, hlineCounts+1);
            cudaFree(d_temp_storage);
        }

            // hlineCounts--; // Removing the last empty row that got add because the last character is '\n'

        // // Creating offset array based on sorted data
        // {
        //     void     *d_temp_storage = NULL;
        //     size_t   temp_storage_bytes = 0;
        //     cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, rowStartsSorted, rowOffsets+1, hlineCounts);
        //     cudaMalloc(&d_temp_storage, temp_storage_bytes);
        //     cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, rowStartsSorted, rowOffsets+1, hlineCounts);            
        //     cudaFree(d_temp_storage);
        // }


        cudaMemset(dCounter,0, sizeof(int));
        forAll(hlineCounts, hornets_nest::countWords {' ', dCounter, d_fileInfo,rowStartsSorted,hlineCounts });
        int hWordCounts = 0;
        cudaMemcpy(&hWordCounts,dCounter,sizeof(int), cudaMemcpyDeviceToHost);
        printf("Word Count : %d \n",hWordCounts);


        gpu::free(rowStarts);
        gpu::free(rowStartsSorted);
        // gpu::free(rowOffsets);


        free(h_fileInfo);
        cudaFree(d_fileInfo);

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  
    printf("%f,", milliseconds/1000.0);             



    cudaFree(dCounter);


    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    // hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    //ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::BfsTopDown2Dynamic>(argc, argv);
    ret = exec<hornets_nest::HornetStaticGraph>(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    // hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

