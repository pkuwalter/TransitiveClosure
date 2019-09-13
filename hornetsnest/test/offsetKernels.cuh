#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest{


struct findAndCount {
	string_t myChar;
	int* count;

    OPERATOR(Vertex& src, Edge& edge) {
    	if(edge.dst_id() == myChar){
    		atomicAdd(count,1);
    		printf("*");
    	}

    }
};

struct findAndCountArray {
	string_t myChar;
	int* count;
	string_t *dataString;

	OPERATOR(int i){
    	if(dataString[i] == myChar){
    		atomicAdd(count,1);
    		// printf("*");
    	}
	}
};

struct findAndStoreLineEnds {
	string_t myChar;
	int* count;
	string_t *dataString;
	soffset_t *positions;

	OPERATOR(int i){
    	if(dataString[i] == myChar){
    		soffset_t myPos = atomicAdd(count,1);
    		positions[myPos] = i;
    		// printf("*");
    	}
	}
};


struct countWords {
	string_t wordSplitter;
	int* count;
	string_t *dataString;
	soffset_t *rowOffsets;
	soffset_t lineCounts;
	// soffset_t *positions;

	OPERATOR(int i){

		soffset_t startPos = rowOffsets[i];
		soffset_t len 		= rowOffsets[i+1]-rowOffsets[i];
		if(len>0)
			len--; //removing '\n' from string

		// if(i>=(lineCounts-10))
		if(i<=(10) || i>=(lineCounts-10))
			printf("%d %d %d\n", i,rowOffsets[i+1],rowOffsets[i]);
		// return;
	    for (int i=0; i<len; i++) {
	    	if(i==0 && dataString[startPos+i] == wordSplitter){
				// atomicAdd(count,1);    		// printf("*");
	    	}else if(i>0){
		    	if(dataString[startPos+i] == wordSplitter && dataString[startPos+i-1]!= wordSplitter){
		    		atomicAdd(count,1);    		// printf("*");
		    	}
	    	}
	    }
	}
};


// struct countWords {
// 	string_t wordSplitter;
// 	string_t lineSplitter;
// 	int* count;
// 	string_t *dataString;
// 	int length;

// 	OPERATOR(int i){
//     	// if(dataString[i] == wordSplitter && (i+1)<(length) && dataString[i+1]!= wordSplitter){
//     	// if(dataString[i] == wordSplitter && (i-1)>0 && dataString[i-1]!= wordSplitter && dataString[i-1]!= lineSplitter){
 
// 	   	if(dataString[i] == lineSplitter){
// 	    		atomicAdd(count,1);
// 	    		// printf("*");
// 	    }

//     	if(dataString[i] == wordSplitter && (i+1)<length && dataString[i-1]!= wordSplitter && dataString[i-1]!= lineSplitter){
//     		atomicAdd(count,1);    		// printf("*");
//     	}
 
// 	}
// };




} // hornets_nest namespace
