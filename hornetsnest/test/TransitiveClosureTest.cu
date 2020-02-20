/**
 * @brief Connected-Component test program
 * @file
 */
#include "Static/TransitiveClosure/transitive_closure.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph(graph::structure_prop::UNDIRECTED);
    CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    HornetGraph hornet_graph(hornet_init);

    TransitiveClosure transClos(hornet_graph);

    Timer<DEVICE> TM;
    TM.start();

    transClos.run(1);

    TM.stop();
    TM.print("Transitive Closure");

    // auto is_correct = cc_multistep.validate();
    // std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    // return !is_correct;
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

