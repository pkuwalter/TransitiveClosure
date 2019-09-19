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

#include "offsetKernels.cuh"


namespace hornets_nest{


    using HornetInit  = ::hornet::HornetInit<string_t,EMPTY,EMPTY,soffset_t>;
    // using HornetDynamicGraph = ::hornet::gpu::Hornet<string_t>;
    using HornetStaticGraph = ::hornet::gpu::HornetStatic<string_t>;

}

#define CHECK_ERROR(str) \
    {cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}


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

        soffset_t *d_rowStarts,*d_rowStartsSorted;//,*rowOffsets;

        gpu::allocate(d_rowStartsSorted,hlineCounts+1);
        gpu::allocate(d_rowStarts,hlineCounts+1);
        // gpu::allocate(rowOffsets,hlineCounts+1);
        cudaMemset(dCounter,0, sizeof(int));
        cudaMemset(d_rowStarts,0, sizeof(int));

        forAll(fsize, hornets_nest::findAndStoreLineEnds {'\n', dCounter, d_fileInfo,d_rowStarts+1});

        cudaMemcpy(&hlineCounts,dCounter,sizeof(int), cudaMemcpyDeviceToHost);
        printf("Line Count : %d \n",hlineCounts);

        // Sorting the starting positions - this is equivalent of an offset array is the values monotonically growing
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_rowStarts, d_rowStartsSorted, hlineCounts+1);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_rowStarts, d_rowStartsSorted, hlineCounts+1);
            cudaFree(d_temp_storage);
        }

        soffset_t *d_rowWordCounter,*d_rowWordOffset;

        gpu::allocate(d_rowWordCounter,hlineCounts+1);
        gpu::allocate(d_rowWordOffset ,hlineCounts+1);

        cudaMemset(d_rowWordCounter,0, sizeof(soffset_t)*(hlineCounts+1));
        cudaMemset(d_rowWordOffset,0, sizeof(soffset_t));

        // Coun the number of words per row
        cudaMemset(dCounter,0, sizeof(int));
        forAll(hlineCounts, hornets_nest::countWords<false> {' ', dCounter, d_fileInfo,d_rowStartsSorted,hlineCounts,d_rowWordCounter, NULL,NULL});
        int hWordCounts = 0;
        cudaMemcpy(&hWordCounts,dCounter,sizeof(int), cudaMemcpyDeviceToHost);
        printf("Word Count : %d \n",hWordCounts);


        // Creating offset of the word counters per row (this will be used for the offset array of the words themselves)
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_rowWordCounter, d_rowWordOffset+1, hlineCounts);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_rowWordCounter, d_rowWordOffset+1, hlineCounts);            
            cudaFree(d_temp_storage);
        }

        cudaMemcpy(&hWordCounts,d_rowWordOffset+hlineCounts,sizeof(soffset_t),cudaMemcpyDeviceToHost);
        printf("Word Count : %d \n",hWordCounts);


        split_t *d_wordSplits;
        soffset_t *d_maxLen,h_maxLen;
        gpu::allocate(d_wordSplits ,hWordCounts);
        gpu::allocate(d_maxLen ,1);

        // Create the compressed sparse representation of the rows (each entry is a separate row)
        forAll(hlineCounts, hornets_nest::countWords<true> {' ', dCounter, d_fileInfo,d_rowStartsSorted,hlineCounts,d_rowWordCounter, d_rowWordOffset,d_wordSplits});


        // ------- Get the size of the largest row in words
        cudaMemset(d_maxLen,0, sizeof(int));
        forAll(hlineCounts, [=] __device__ (int row){ atomicMax(d_maxLen,d_rowWordCounter[row]); } );
        cudaMemcpy(&h_maxLen,d_maxLen,sizeof(soffset_t),cudaMemcpyDeviceToHost);
        printf("Largest Line in Words is : %d \n",h_maxLen);
        cudaMemset(d_maxLen,0, sizeof(int));
        forAll(hlineCounts, [=] __device__ (int row){ atomicMax(d_maxLen,d_rowWordOffset[row+1]-d_rowWordOffset[row]); });
        cudaMemcpy(&h_maxLen,d_maxLen,sizeof(soffset_t),cudaMemcpyDeviceToHost);
        printf("Largest Line in Words is : %d \n",h_maxLen);


        // -------
	    soffset_t *d_columnStringSize,*d_columnStringOffset;

        gpu::allocate(d_columnStringSize, hlineCounts+1);
        gpu::allocate(d_columnStringOffset, hlineCounts+1);


        void     *d_temp_storage_string_offset = NULL; size_t   temp_storage_bytes_string_offset = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage_string_offset, temp_storage_bytes_string_offset, d_columnStringSize, d_columnStringOffset+1, hlineCounts);
        cudaMalloc(&d_temp_storage_string_offset, temp_storage_bytes_string_offset);



		soffset_t sumWords=0;
        soffset_t *d_wordColumnSize,h_wordColumnSize;
	    soffset_t *d_wordPerColumn,h_wordPerColumn;

        gpu::allocate(d_wordPerColumn ,1);
        gpu::allocate(d_wordColumnSize ,1);

        // Going through all the columns of the data
        for(soffset_t c=0; c<=h_maxLen;c++){

	        cudaMemset(d_wordPerColumn,0, sizeof(soffset_t));
	        cudaMemset(d_wordColumnSize,0, sizeof(soffset_t));

	        forAll(hlineCounts, hornets_nest::countWordPerColumn {d_rowWordCounter,d_wordPerColumn,d_wordColumnSize,c,d_rowWordOffset,d_wordSplits,d_columnStringSize});

	        cudaMemcpy(&h_wordPerColumn,d_wordPerColumn,sizeof(soffset_t),cudaMemcpyDeviceToHost);
	        cudaMemcpy(&h_wordColumnSize,d_wordColumnSize,sizeof(soffset_t),cudaMemcpyDeviceToHost);

	        string_t* d_columnStringData;

	        gpu::allocate(d_columnStringData,sizeof(d_wordColumnSize));

	        cudaMemset(d_columnStringOffset,0, sizeof(soffset_t));
	        cub::DeviceScan::InclusiveSum(d_temp_storage_string_offset, temp_storage_bytes_string_offset, d_columnStringSize, d_columnStringOffset+1, hlineCounts);

	        forAll(hlineCounts, hornets_nest::copyWordsToColumn {d_rowWordCounter,c,d_fileInfo, d_rowStartsSorted,
	        	d_rowWordOffset,d_wordSplits,d_columnStringOffset,d_columnStringData,hlineCounts});

	        gpu::free(d_columnStringData);
	        // printf("(%d, %d, %d), ", c, h_wordPerColumn,h_wordColumnSize);
	        sumWords+=h_wordPerColumn;
        }
        gpu::free(d_wordColumnSize);
        gpu::free(d_wordPerColumn);

        cudaFree(d_temp_storage_string_offset);

        printf("\n");

        printf("The sum of the words is %d\n",sumWords);

		gpu::free(d_columnStringSize);
		gpu::free(d_columnStringOffset);        


        gpu::free(d_maxLen);
        gpu::free(d_wordSplits);
        gpu::free(d_rowWordCounter);
        gpu::free(d_rowWordOffset);


        gpu::free(d_rowStarts);
        gpu::free(d_rowStartsSorted);
        // gpu::free(rowOffsets);


        free(h_fileInfo);
        cudaFree(d_fileInfo);

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  
    printf("%f,", milliseconds/1000.0);             
    printf("\n");



    cudaFree(dCounter);


    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    //ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::BfsTopDown2Dynamic>(argc, argv);
    ret = exec<hornets_nest::HornetStaticGraph>(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

