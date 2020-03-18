/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

template <typename HornetGraph, typename BFS>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;


    graph::GraphStd<vid_t, eoff_t> graph(DIRECTED );
    CommandLineParam cmd(graph, argc, argv,false);


    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    Timer<DEVICE> TM;
    HornetGraph hornet_graph(hornet_init);

    BFS bfs_top_down(hornet_graph);

    vid_t root = graph.max_out_degree_id();
    // if (argc==3)
    //     root = atoi(argv[2]);
    int numberRoots = 10;
    if (argc>=3)
      numberRoots = atoi(argv[2]);

    int alg = 0;
    if (argc>=4)
      alg = atoi(argv[3]);

  printf("Alg is %d\n",alg);

    std::cout << "My root is " << root << std::endl;


    cudaProfilerStart();
    TM.start();
    for(int i=0; i<numberRoots; i++){
        bfs_top_down.reset();
        bfs_top_down.set_parameters((root+i)%graph.nV());
        bfs_top_down.runAlg(alg);
        std::cout << "Number of levels is : " << bfs_top_down.getLevels() << std::endl;

    }

    TM.stop();
    cudaProfilerStop();
    TM.print("TopDown2");


    auto is_correct = bfs_top_down.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    //ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::BfsTopDown2Dynamic>(argc, argv);
    ret = exec<hornets_nest::HornetStaticGraph,  hornets_nest::BfsTopDown2Static >(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

