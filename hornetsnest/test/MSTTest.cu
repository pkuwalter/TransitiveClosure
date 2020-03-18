/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/MinimalSpanningTree/minimal-spanning-tree.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

template <typename HornetGraph, typename MST>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);


    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());


    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    std::vector<wgt0_t> edge_meta_0(graph.nE(), 0);
    hornet_init.insertEdgeData(edge_meta_0.data());
    HornetDynamicGraph hornet_gpu(hornet_init);


    TM.stop();
    cudaProfilerStop();
    TM.print("Initilization Time:");

    MST mst_new(hornet_gpu);


    cudaProfilerStart();
    TM.start();
    mst_new.reset();
    // mst_new.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("Parallel MST");

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::MinimalSpanningTreeDynamic>(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

