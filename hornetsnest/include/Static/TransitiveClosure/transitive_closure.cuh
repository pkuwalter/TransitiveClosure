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
protected:
   // triangle_t* triPerVertex { nullptr };

    trans_t* d_CountNewEdges;

    vid_t* d_src { nullptr };
    vid_t* d_dest { nullptr };
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
        for (vid_t i = 0; i < (size-1); i++) {
            vid_t min_idx=i;

            for(vid_t j=i+1; j<(size); j++){
                if(vertex.neighbor_ptr()[j]<vertex.neighbor_ptr()[min_idx])
                    min_idx=j;
                // if (vertex.neighbor_ptr()[j]==vertex.neighbor_ptr()[j-1])
                //     printf("*");
            }
            vid_t temp = vertex.neighbor_ptr()[i];
            vertex.neighbor_ptr()[i] = vertex.neighbor_ptr()[min_idx];
            vertex.neighbor_ptr()[min_idx] = temp;
        }
 
    }
};



template <bool countOnly>
struct OPERATOR_AdjIntersectionCountBalanced {
    trans_t* d_CountNewEdges;
    vid_t* d_src ;
    vid_t* d_dest;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
        int count = 0;
        if (!FLAG) {
            int comp_equals, comp1, comp2, ui_bound, vi_bound;
            //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
            while (vi_begin <= vi_end && ui_begin <= ui_end) {
                comp_equals = (*ui_begin == *vi_begin);
                if(countOnly){
                    count += 1-comp_equals;
                    // if(u.id()==0 || v.id()==0)
                    //     printf("%d %d %d %d\n", u.id(), v.id(), *ui_begin, *vi_begin);
                }
                else{
                    if(!comp_equals){
                        vid_t second = *vi_begin;
                        if (*ui_begin<*vi_begin)
                            second=*ui_begin;

                        trans_t pos = atomicAdd(d_CountNewEdges, 2);
                        d_src[pos]  = u.id();
                        d_dest[pos] = second;
                        d_src[pos+1]  = v.id();
                        d_dest[pos+1] = second;

                        // if(u.id()==0 || v.id()==0)
                        //     printf("%d %d %d\n", u.id(), v.id(), second)

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
        } else {
            vid_t vi_low, vi_high, vi_mid;
            while (ui_begin <= ui_end) {
                auto search_val = *ui_begin;
                vi_low = 0;
                vi_high = vi_end-vi_begin;
                bool earlyBreak=false;
                while (vi_low <= vi_high) {
                    vi_mid = (vi_low+vi_high)/2;
                    auto comp = (*(vi_begin+vi_mid) - search_val);
                    if (!comp) {
                        // count += 1;
                        earlyBreak=true;
                        break;
                    }
                    if (comp > 0) {
                        vi_high = vi_mid-1;
                    } else if (comp < 0) {
                        vi_low = vi_mid+1;
                    }
                }
                if(earlyBreak==false){
                    // printf("$$$\n");
                    if(countOnly){
                        count++; // If the value has been found. We don't want to add an edge
                    }else{
                        trans_t pos = atomicAdd(d_CountNewEdges, 2);
                        d_src[pos]  = u.id();
                        d_dest[pos] = search_val;
                        d_src[pos+1]  = v.id();
                        d_dest[pos+1] = search_val;
                    }
                }
                ui_begin += 1;
            }
        }
        if(count>0){
            if(countOnly){
                atomicAdd(d_CountNewEdges, count);
            }
        }
    }
};


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
        h_batchSize *=2;

        cudaMemset(d_CountNewEdges,0,sizeof(trans_t));
        gpu::allocate(d_src, h_batchSize );
        gpu::allocate(d_dest, h_batchSize);


        // cudaMemset(d_src,0,sizeof(vid_t)*h_batchSize);
        // cudaMemset(d_dest,0,sizeof(vid_t)*h_batchSize);


        forAllAdjUnions(hornet, OPERATOR_AdjIntersectionCountBalanced<false> { d_CountNewEdges, d_src, d_dest }, WORK_FACTOR);
        cudaDeviceSynchronize();


        UpdatePtr ptr(h_batchSize, d_src, d_dest);
        Update batch_update(ptr);
        hornet.insert(batch_update,true,true);
        cudaDeviceSynchronize();
        printf("New batch size is %lld and HornetSize %d \n", h_batchSize, hornet.nE());

        sortHornet();

        gpu::free(d_src);
        gpu::free(d_dest);

        iterations++;
        if(iterations==10)
            break;
    }
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