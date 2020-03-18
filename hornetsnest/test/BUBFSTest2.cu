/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BUBreadthFirstSearch/BottomUpBFS.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;
    using vid_t = int;
    using dst_t = int;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    // graph::GraphStd<vid_t, eoff_t> graph;
    graph::GraphStd<vid_t, eoff_t> graph(DIRECTED | ENABLE_INGOING);
    CommandLineParam cmd(graph, argc, argv,false);


    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());


    HornetInit hornet_init_inverse(graph.nV(), graph.nE(),
                                   graph.csr_in_offsets(),
                                   graph.csr_in_edges());



    HornetGraph hornet_graph_inv(hornet_init_inverse);
    HornetGraph hornet_graph(hornet_init);


    BfsBottomUp2 bfs_bottom_up(hornet_graph, hornet_graph_inv);
    BfsBottomUp2 bfs_top_down(hornet_graph, hornet_graph_inv);
 
	vid_t root = graph.max_out_degree_id();
	// if (argc==3)
	//   root = atoi(argv[2]);

    int numberRoots = 10;
    if (argc==3)
      numberRoots = atoi(argv[2]);

    Timer<DEVICE> TM;

    cudaProfilerStart();
    TM.start();
    for(int i=0; i<numberRoots; i++){
        bfs_top_down.reset();
        bfs_bottom_up.set_parameters((root+i)%graph.nV());
        bfs_top_down.set_parameters((root+i)%graph.nV());
     
        bfs_bottom_up.run(hornet_graph_inv);

    }

    TM.stop();
    cudaProfilerStop();
    TM.print("Direction-Optimizing");


    auto is_correct = bfs_bottom_up.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
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

