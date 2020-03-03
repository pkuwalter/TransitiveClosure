#pragma once

#include "HornetAlg.hpp"
#include <Graph/GraphStd.hpp>


#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
using namespace timer;

namespace hornets_nest {

//using triangle_t = int;
using trans_t = unsigned long long;
using vid_t = int;

using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit  = ::hornet::HornetInit<vid_t>;

using UpdatePtr   = ::hornet::BatchUpdatePtr<vid_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vid_t>;

//==============================================================================

class TransitiveClosure : public StaticAlgorithm<HornetGraph> {
public:
    TransitiveClosure(HornetGraph& hornet);
    ~TransitiveClosure();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void run(const int WORK_FACTOR);
    void init();
    void sortHornet();

    void cleanGraph();

protected:
   // triangle_t* triPerVertex { nullptr };

    trans_t* d_CountNewEdges;

    vid_t* d_src { nullptr };
    vid_t* d_dest { nullptr };
    vid_t* d_srcOut { nullptr };
    vid_t* d_destOut { nullptr };
    // batch_t* d_batchSize { nullptr };

};

//==============================================================================

} // namespace hornets_nest


#include <cuda.h>
#include <cuda_runtime.h>

namespace hornets_nest {

TransitiveClosure::TransitiveClosure(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet){       
    init();
}

TransitiveClosure::~TransitiveClosure(){
    release();
}



struct SimpleBubbleSort {

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();

        // if(vertex.id()<5)
        //     printf("%d %d\n", vertex.id(),vertex.degree());

        degree_t size = vertex.degree();
        if(size<=1)
            return;

        // if(src==250){
        //     for (vid_t i = 0; i < (size); i++) {
        //         printf("%d ", vertex.neighbor_ptr()[i]);
        //     }
        //     printf("\n");
        // }
// (250,0,1,5,252,252)

        for (vid_t i = 0; i < (size-1); i++) {
            vid_t min_idx=i;

            for(vid_t j=(i+1); j<(size); j++){
                if(vertex.neighbor_ptr()[j]<vertex.neighbor_ptr()[min_idx]){
                    min_idx=j;
                }
                if (vertex.neighbor_ptr()[j]==vertex.neighbor_ptr()[j-1]){
                    // printf("(%d,%d,%d,%d,%d,%d)\n",src,i,j,size,vertex.neighbor_ptr()[j],vertex.neighbor_ptr()[j-1]);
                }
                //     printf("*");
            }
            vid_t temp = vertex.neighbor_ptr()[i];
            vertex.neighbor_ptr()[i] = vertex.neighbor_ptr()[min_idx];
            vertex.neighbor_ptr()[min_idx] = temp;
        }
        //  if(src==250){
        //     for (vid_t i = 0; i < (size); i++) {
        //         printf("%d ", vertex.neighbor_ptr()[i]);
        //     }
        //     printf("\n");
        // }

    }
};



template <bool countOnly>
struct OPERATOR_AdjIntersectionCountBalanced {
    trans_t* d_CountNewEdges;
    vid_t* d_src ;
    vid_t* d_dest;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
        int count = 0;

        if(u.id()==v.id()){
            // printf("@");
            return;
        }

        bool vSrc=false;
        // vid_t src = u.id();
        // vid_t dest = v.id();
        if(FLAG&2){
            vSrc=true;
            // src = v.id();
            // dest = u.id();
            // printf("^");
        }


        if (!FLAG) {
            int comp_equals, comp1, comp2, ui_bound, vi_bound;
            //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
            while (vi_begin <= vi_end && ui_begin <= ui_end) {
                comp_equals = (*ui_begin == *vi_begin);
                if(!comp_equals){

                    if(!vSrc && *ui_begin > *vi_begin && *vi_begin != u.id() && *vi_begin != v.id()){
                        if(countOnly){
                            count++;
                        }else{
                            trans_t pos = atomicAdd(d_CountNewEdges, 1);
                            d_src[pos]  = u.id();
                            d_dest[pos] = *vi_begin;
                        }
                    }else if(vSrc && *ui_begin < *vi_begin && *ui_begin !=v.id() && *ui_begin != u.id()){
                        if(countOnly){
                            count++;
                        }else{
                            trans_t pos = atomicAdd(d_CountNewEdges, 1);
                            d_src[pos]  = v.id();
                            d_dest[pos] = *ui_begin;
                        }
                    }
                    
                }

                // count += comp_equals;
                comp1 = (*ui_begin >= *vi_begin);
                comp2 = (*ui_begin <= *vi_begin);
                ui_bound = (ui_begin == ui_end);
                vi_bound = (vi_begin == vi_end);
                // early termination
                if ((ui_bound && comp2) || (vi_bound && comp1))
                    break;
                if ((comp1 && !vi_bound) || ui_bound)
                    vi_begin += 1;
                if ((comp2 && !ui_bound) || vi_bound)
                    ui_begin += 1;
            }
            while(!vSrc && vi_begin <= vi_end){
                if(*vi_begin != u.id()){
                    if(countOnly){
                        count++;
                    }else{
                        trans_t pos = atomicAdd(d_CountNewEdges, 1);
                        d_src[pos]  = u.id();
                        d_dest[pos] = *vi_begin;
                    }
                }
                vi_begin +=1;
            }
            while(vSrc && ui_begin <= ui_end){
                if(*ui_begin != v.id()){
                    if(countOnly){
                        count++;
                    }else{
                        trans_t pos = atomicAdd(d_CountNewEdges, 1);
                        d_src[pos]  = v.id();
                        d_dest[pos] = *ui_begin;
                    }
                }
                ui_begin +=1;
            }


        } else {
            // if((ui_end!=ui_begin) && (vi_end!=vi_begin)){
            //     printf("This shouldn't happen %d %d\n", ui_end-ui_begin,vi_end-vi_begin);
            // }
            return;
            // vid_t vi_low, vi_high, vi_mid;
            // while (ui_begin <= ui_end) {
            //     auto search_val = *ui_begin;
            //     vi_low = 0;
            //     vi_high = vi_end-vi_begin;
            //     bool earlyBreak=false;
            //     while (vi_low <= vi_high) {
            //         vi_mid = (vi_low+vi_high)/2;
            //         auto comp = (*(vi_begin+vi_mid) - search_val);
            //         if (!comp) {
            //             // count += 1;
            //             earlyBreak=true;
            //             break;
            //         }
            //         if (comp > 0) {
            //             vi_high = vi_mid-1;
            //         } else if (comp < 0) {
            //             vi_low = vi_mid+1;
            //         }
            //     }
            //     if(earlyBreak==false){
            //         // printf("$$$\n");
            //         if(countOnly){
            //             count++; // If the value has been found. We don't want to add an edge
            //         }else{
            //             trans_t pos = atomicAdd(d_CountNewEdges, 2);
            //             d_src[pos]  = u.id();
            //             d_dest[pos] = search_val;
            //             d_src[pos+1]  = v.id();
            //             d_dest[pos+1] = search_val;
            //         }
            //     }
            //     ui_begin += 1;
            // }
        }
        if(count>0){
            if(countOnly){
                atomicAdd(d_CountNewEdges, count);
            }
        }
    }
};


__global__ void filterSortedBatch(trans_t originalBatchSize, trans_t* newBatchSize, 
    vid_t* srcSorted, vid_t* destSorted,
    vid_t* srcFiltered, vid_t* destFiltered){

    trans_t i = blockIdx.x*blockDim.x + threadIdx.x;
    trans_t stride = blockDim.x*gridDim.x; 
    // if(i==0)
    //     printf("stride = %llu \n",stride);

    for (; i < originalBatchSize; i+=stride){
        if(i==0){
            trans_t pos = atomicAdd(newBatchSize,1);
            srcFiltered[pos]  = srcSorted[0];
            destFiltered[pos] = destSorted[0];
        }else{
            if((srcSorted[i]!=srcSorted[i-1]) || (srcSorted[i]==srcSorted[i-1] && destSorted[i]!=destSorted[i-1])){
                trans_t pos = atomicAdd(newBatchSize,1);
                srcFiltered[pos] = srcSorted[i];
                destFiltered[pos] = destSorted[i];
            }else if(srcSorted[i]==destSorted[i]){
                printf("$");
            }
        }
    }
}

template <bool countOnly>
struct findDuplicatesForRemoval {                  //deterministic
    trans_t* newBatchSize;
    vid_t* srcDelete;
    vid_t* destDelete;
    OPERATOR(Vertex& vertex) {

        degree_t size = vertex.degree();
        if(size<=1)
            return;

        // if(vertex.id()==250)
        //     printf("*%d\n",vertex.neighbor_ptr()[0]);

        for (vid_t i = 1; i < (size); i++) {
        //     if(vertex.id()==250)
        //         printf("*%d\n",vertex.neighbor_ptr()[i]);


            if(vertex.neighbor_ptr()[i]==vertex.neighbor_ptr()[i-1]){
                if(countOnly){
                    atomicAdd(newBatchSize,1);
                }else{
                    trans_t pos = atomicAdd(newBatchSize,1);
                    srcDelete[pos] = vertex.id();
                    destDelete[pos] = vertex.neighbor_ptr()[i];
                }
            }
        }
    }
};
//-------
    

void TransitiveClosure::reset(){

    cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
    sortHornet();
}

void TransitiveClosure::run() {
    // forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced { triPerVertex }, 1);
}

void TransitiveClosure::run(const int WORK_FACTOR=1){

    int iterations=0;
    while(true){

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced<true> { d_CountNewEdges, d_src, d_dest }, WORK_FACTOR);

        trans_t h_batchSize;
        cudaMemcpy(&h_batchSize,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);

        if(h_batchSize==0){
            break;
        }
        // h_batchSize *=2;
        printf("First  - New batch size is %lld and HornetSize %d \n", h_batchSize, hornet.nE());


        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        // gpu::allocate(d_src, h_batchSize);
        // gpu::allocate(d_dest, h_batchSize);

        cudaMallocManaged(&d_src, h_batchSize*sizeof(trans_t));
        cudaMallocManaged(&d_dest, h_batchSize*sizeof(trans_t));
        cudaMallocManaged(&d_srcOut, h_batchSize*sizeof(trans_t));
        cudaMallocManaged(&d_destOut, h_batchSize*sizeof(trans_t));

        // gpu::allocate(d_src, h_batchSize);
        // gpu::allocate(d_dest, h_batchSize);


        forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced<false> { d_CountNewEdges, d_src, d_dest }, WORK_FACTOR);
        cudaDeviceSynchronize();
        trans_t unFilterBatchSize = h_batchSize;
        vid_t* temp;

        if(1){
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                d_dest, d_destOut, d_src, d_srcOut, h_batchSize);
            // Allocate temporary storage
            cudaMallocManaged(&d_temp_storage, temp_storage_bytes);
            // Run sorting operation


            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                d_dest, d_destOut, d_src, d_srcOut, h_batchSize);
            cudaDeviceSynchronize();
            temp = d_dest; d_dest=d_destOut; d_destOut=temp;
            temp = d_src; d_src=d_srcOut; d_srcOut=temp;

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                 d_src, d_srcOut, d_dest, d_destOut, h_batchSize);
            cudaDeviceSynchronize();
            temp = d_dest; d_dest=d_destOut; d_destOut=temp;
            temp = d_src; d_src=d_srcOut; d_srcOut=temp;

            gpu::free(d_temp_storage);

        }else{
            thrust::stable_sort_by_key(thrust::device, d_dest, d_dest + h_batchSize, d_src);
            thrust::stable_sort_by_key(thrust::device, d_src, d_src + h_batchSize, d_dest);            
            cudaDeviceSynchronize();
        }

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        filterSortedBatch<<<1024,256>>>(unFilterBatchSize,d_CountNewEdges,d_src,d_dest,d_srcOut,d_destOut);
        cudaDeviceSynchronize();

        trans_t h_batchSizeNew;

        cudaMemcpy(&h_batchSizeNew,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);
        temp = d_dest; d_dest=d_destOut; d_destOut=temp;
        temp = d_src; d_src=d_srcOut; d_srcOut=temp;

        printf("Intermediate - Before  %lld and after %lld\n", h_batchSize,h_batchSizeNew);


        gpu::free(d_srcOut);
        gpu::free(d_destOut);

        if(!h_batchSizeNew){
            break;
        }

        UpdatePtr ptr(h_batchSizeNew, d_src, d_dest);
        Update batch_update(ptr);
        hornet.insert(batch_update,false,false);
        cudaDeviceSynchronize();
        printf("Second - New batch size is %lld and HornetSize %d \n", h_batchSizeNew, hornet.nE());

        sortHornet();


        gpu::free(d_src);
        gpu::free(d_dest);

        iterations++;

        cleanGraph();
        // if(iterations==1)
        //     break;
    }
}

void TransitiveClosure::cleanGraph(){

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));


        forAllVertices(hornet, findDuplicatesForRemoval<true>{d_CountNewEdges, d_src, d_dest});

        trans_t h_batchSize;
        cudaMemcpy(&h_batchSize,d_CountNewEdges, sizeof(trans_t),cudaMemcpyDeviceToHost);

        if(!h_batchSize)
            return;

        cudaMallocManaged(&d_src, h_batchSize*sizeof(trans_t));
        cudaMallocManaged(&d_dest, h_batchSize*sizeof(trans_t));

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));


        forAllVertices(hornet, findDuplicatesForRemoval<false>{d_CountNewEdges, d_src, d_dest});
        printf("Number of duplicates in initial graph is: %lld\n",h_batchSize);

        UpdatePtr ptr(h_batchSize, d_src, d_dest);
        Update batch_update(ptr);
        hornet.erase(batch_update);

        cudaDeviceSynchronize();

        sortHornet();

        gpu::free(d_src);
        gpu::free(d_dest);

}


void TransitiveClosure::release(){
    gpu::free(d_CountNewEdges);
    d_CountNewEdges = nullptr;
}

void TransitiveClosure::init(){
    gpu::allocate(d_CountNewEdges, 1);
    reset();
}


void TransitiveClosure::sortHornet(){
    forAllVertices(hornet, SimpleBubbleSort {});
}


} // namespace hornets_nest