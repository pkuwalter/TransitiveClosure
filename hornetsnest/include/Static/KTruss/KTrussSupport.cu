// #include "Static/KTruss/KTruss.cuh"


namespace hornets_nest {


__device__ __forceinline__
void initialize(degree_t diag_id,
                degree_t u_len,
                degree_t v_len,
                vert_t* __restrict__ u_min,
                vert_t* __restrict__ u_max,
                vert_t* __restrict__ v_min,
                vert_t* __restrict__ v_max,
                int*   __restrict__ found) {
    if (diag_id == 0) {
        *u_min = *u_max = *v_min = *v_max = 0;
        *found = 1;
    }
    else if (diag_id < u_len) {
        *u_min = 0;
        *u_max = diag_id;
        *v_max = diag_id;
        *v_min = 0;
    }
    else if (diag_id < v_len) {
        *u_min = 0;
        *u_max = u_len;
        *v_max = diag_id;
        *v_min = diag_id - u_len;
    }
    else {
        *u_min = diag_id - v_len;
        *u_max = u_len;
        *v_min = diag_id - u_len;
        *v_max = v_len;
    }
}

__device__ __forceinline__
void workPerThread(degree_t uLength,
                   degree_t vLength,
                   int threadsPerIntersection,
                   int threadId,
                   int* __restrict__ outWorkPerThread,
                   int* __restrict__ outDiagonalId) {
  int      totalWork = uLength + vLength;
  int  remainderWork = totalWork % threadsPerIntersection;
  int  workPerThread = totalWork / threadsPerIntersection;

  int longDiagonals  = threadId > remainderWork ? remainderWork : threadId;
  int shortDiagonals = threadId > remainderWork ? threadId - remainderWork : 0;

  *outDiagonalId     = (workPerThread + 1) * longDiagonals +
                        workPerThread * shortDiagonals;
  *outWorkPerThread  = workPerThread + (threadId < remainderWork);
}

__device__ __forceinline__
void bSearch(unsigned found,
             degree_t    diagonalId,
             const vert_t*  __restrict__ uNodes,
             const vert_t*  __restrict__ vNodes,
             const degree_t*  __restrict__ uLength,
             vert_t* __restrict__ outUMin,
             vert_t* __restrict__ outUMax,
             vert_t* __restrict__ outVMin,
             vert_t* __restrict__ outVMax,
             vert_t* __restrict__ outUCurr,
             vert_t* __restrict__ outVCurr) {
    vert_t length;
    while (!found){
        *outUCurr = (*outUMin + *outUMax) >> 1;
        *outVCurr = diagonalId - *outUCurr;
        if (*outVCurr >= *outVMax){
            length = *outUMax - *outUMin;
            if (length == 1){
                found = 1;
                continue;
            }
        }

        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (comp1 && !comp2)
            found = 1;
        else if (comp1){
            *outVMin = *outVCurr;
            *outUMax = *outUCurr;
        }
        else{
            *outVMax = *outVCurr;
            *outUMin = *outUCurr;
        }
    }

    if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
            *outUCurr > 0 && *outUCurr < *uLength - 1)
    {
        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (!comp1 && !comp2)
        {
            (*outUCurr)++;
            (*outVCurr)--;
        }
    }
}

__device__ __forceinline__
int fixStartPoint(degree_t uLength, degree_t vLength,
                  vert_t* __restrict__ uCurr,
                  vert_t* __restrict__ vCurr,
                  const vert_t* __restrict__ uNodes,
                  const vert_t* __restrict__ vNodes) {

    unsigned uBigger = (*uCurr > 0) && (*vCurr < vLength) &&
                       (uNodes[*uCurr - 1] == vNodes[*vCurr]);
    unsigned vBigger = (*vCurr > 0) && (*uCurr < uLength) &&
                       (vNodes[*vCurr - 1] == uNodes[*uCurr]);
    *uCurr += vBigger;
    *vCurr += uBigger;
    return uBigger + vBigger;
}


__device__ __forceinline__
void indexBinarySearch(vert_t* data, vert_t arrLen, vert_t key, int& pos) {
    int low = 0;
    int high = arrLen - 1;
    while (high >= low)
    {
        int middle = (low + high) / 2;
        if (data[middle] == key)
        {
            pos = middle;
            return;
        }
        if (data[middle] < key)
            low = middle + 1;
        if (data[middle] > key)
            high = middle - 1;
    }
}

template<typename HornetDevice>
__device__ __forceinline__
void intersectCount(const HornetDevice& hornet,
        degree_t uLength, degree_t vLength,
        const vert_t*  __restrict__ uNodes,
        const vert_t*  __restrict__ vNodes,
        vert_t*  __restrict__ uCurr,
        vert_t*  __restrict__ vCurr,
        int*    __restrict__ workIndex,
        const int*    __restrict__ workPerThread,
        triangle_t*    __restrict__ triangles,
        int found,
        triangle_t*  __restrict__ outPutTriangles,
        vert_t src, vert_t dest,
    vert_t u, vert_t v) {
    if (*uCurr < uLength && *vCurr < vLength) {
        int comp;
        int vmask;
        int umask;
        while (*workIndex < *workPerThread)
        {
            vmask = umask = 0;
            comp = uNodes[*uCurr] - vNodes[*vCurr];

            *triangles += (comp == 0);

            *uCurr += (comp <= 0 && !vmask) || umask;
            *vCurr += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0 && !umask && !vmask) + 1;

            if (*vCurr >= vLength || *uCurr >= uLength)
                break;
        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && found);
    }
}

template<typename HornetDevice>
__device__ __forceinline__
triangle_t count_triangles(const HornetDevice& hornet,
                           vert_t u,
                           const vert_t* __restrict__ u_nodes,
                           degree_t u_len,
                           vert_t v,
                           const vert_t* __restrict__ v_nodes,
                           degree_t v_len,
                           int   threads_per_block,
                           volatile triangle_t* __restrict__ firstFound,
                           int    tId,
                           triangle_t* __restrict__ outPutTriangles,
                           const vert_t*      __restrict__ uMask,
                           const vert_t*      __restrict__ vMask,
                           triangle_t multiplier,
                           vert_t      src,
                           vert_t      dest) {

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0;
    int            found = 0;
    vert_t u_min, u_max, v_min, v_max, u_curr, v_curr;

    firstFound[tId] = 0;

    if (work_per_thread > 0) {
        // For the binary search, we are figuring out the initial poT of search.
        initialize(diag_id, u_len, v_len, &u_min, &u_max,
                   &v_min, &v_max, &found);
        u_curr = 0;
        v_curr = 0;

        bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                &v_min, &v_max, &u_curr, &v_curr);

        int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr,
                                u_nodes, v_nodes);
        work_index += sum;
        if (tId > 0)
           firstFound[tId - 1] = sum;
        triangles += sum;
        intersectCount
            (hornet, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
            &work_index, &work_per_thread, &triangles, firstFound[tId],
            outPutTriangles, src, dest, u, v);

    }
    return triangles;
}



__device__ __forceinline__
void workPerBlock(vert_t numVertices,
                  vert_t* __restrict__ outMpStart,
                  vert_t* __restrict__ outMpEnd,
                  int blockSize) {
    vert_t       verticesPerMp = numVertices / gridDim.x;
    vert_t     remainderBlocks = numVertices % gridDim.x;
    vert_t   extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks
                                                               : blockIdx.x;
    vert_t regularVertexBlocks = (blockIdx.x > remainderBlocks) ?
                                    blockIdx.x - remainderBlocks : 0;

    vert_t mpStart = (verticesPerMp + 1) * extraVertexBlocks +
                     verticesPerMp * regularVertexBlocks;
    *outMpStart   = mpStart;
    *outMpEnd     = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}


//==============================================================================
//==============================================================================


template<typename HornetDevice>
__global__
void devicecuHornetKTruss(HornetDevice hornet,
                           triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           HostDeviceVar<KTrussData> hd_data) {
    KTrussData* __restrict__ devData = hd_data.ptr();
    vert_t nv = hornet.nV();
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vert_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ vert_t      firstFound[1024];

    vert_t     adj_offset = tx >> shifter;
    vert_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vert_t src = this_mp_start; src < this_mp_stop; src++) {
        auto vertex = hornet.vertex(src);
        vert_t srcLen = vertex.degree();

        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            vert_t dest = vertex.edge(k).dst_id();
            degree_t destLen = hornet.vertex(dest).degree();
            // if (dest < src) //opt
            //     continue;   //opt

            bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
            if (avoidCalc)
                continue;

            bool sourceSmaller = srcLen < destLen;
            vert_t        small = sourceSmaller ? src : dest;
            vert_t        large = sourceSmaller ? dest : src;
            degree_t    small_len = sourceSmaller ? srcLen : destLen;
            degree_t    large_len = sourceSmaller ? destLen : srcLen;

            const vert_t* small_ptr = hornet.vertex(small).neighbor_ptr();
            const vert_t* large_ptr = hornet.vertex(large).neighbor_ptr();

            triangle_t triFound = count_triangles
                (hornet, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, (triangle_t*)firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);


            int pos = hd_data().offset_array[src] + k;
            atomicAdd(hd_data().triangles_per_edge + pos,triFound);


            // atomicAdd(outPutTriangles+src,triFound);
            // atomicAdd(outPutTriangles+dest,triFound);

        }
    }
}



//==============================================================================

void kTrussOneIteration(HornetGraph& hornet,
                        triangle_t* __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data) {
    devicecuHornetKTruss <<< thread_blocks, blockdim >>>
        (hornet.device(), output_triangles, threads_per_block,
         number_blocks, shifter, hd_data);

}

} // namespace hornets_nest
