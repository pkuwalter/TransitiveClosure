#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

// const bool _FORCE_SOA = true;

using triangle_t = vert_t;

using HornetGraph = hornet::gpu::Hornet<vert_t>;
// using HornetGraph = gpu::Hornet<vert_t>;



using HornetInit  = ::hornet::HornetInit<vert_t>;

using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vert_t>;

// using wgt0_t = int;

// using Init = hornet::HornetInit<vert_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;
// using KTrussHornet = hornet::gpu::Hornet<vert_t, hornet::EMPTY, hornet::TypeList<wgt0_t>>;


struct KTrussData {
    int max_K;

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* is_active;
    int* offset_array;
    int* triangles_per_edge;
    int* triangles_per_vertex;

    vert_t* src;
    vert_t* dst;
    int*    counter;
    int*    active_vertices;

    TwoLevelQueue<vert_t> active_queue; // Stores all the active vertices

    int full_triangle_iterations;

    vert_t nv;
    off_t ne;                  // undirected-edges
    off_t num_edges_remaining; // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class KTruss : public StaticAlgorithm<HornetGraph> {
public:
    KTruss(HornetGraph& hornet);
    ~KTruss();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    //--------------------------------------------------------------------------
    void setInitParameters(int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    bool findTrussOfK(bool& stop);
    void runForK(int max_K);

    void createOffSetArray();
    void copyOffsetArrayHost(const vert_t* host_offset_array);
    void copyOffsetArrayDevice(vert_t* device_offset_array);
    void resetEdgeArray();
    void resetVertexArray();

    vert_t getIterationCount();
    vert_t getMaxK();

    void sortHornet();

private:
    HostDeviceVar<KTrussData> hd_data;

    vert_t originalNE;
    vert_t originalNV;
};

#define CHECK_ERROR(str) \
    {cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout); exit(0);}}

} // namespace hornets_nest


#include "KTruss.impl.cuh"