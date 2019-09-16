#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest{

using string_t = char;
using soffset_t = int;

typedef struct {
	soffset_t start;
	soffset_t len;
} split_t;



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


template <bool storeOffsets>
struct countWords {
	string_t wordSplitter;
	int* count;
	string_t *dataString;
	soffset_t *rowDataOffsets;
	soffset_t lineCounts;
	soffset_t *rowWordCounter;
	soffset_t *rowWordOffsets;
	split_t* wordSplits;

	OPERATOR(int row){

		soffset_t startPos = rowDataOffsets[row];
		soffset_t len 		= rowDataOffsets[row+1]-rowDataOffsets[row];
		if(len>0)
			len--; //removing '\n' from string

		soffset_t words=0;
		soffset_t l=0;
		while (l<len){
			//Removing all word spliters before word
			while(dataString[startPos+l]==wordSplitter && l<len)
				l++;
			if(l==len)
				break;
			soffset_t start = l;
			while(dataString[startPos+l]!=wordSplitter && l<len)	
				l++;
			soffset_t wordLength = l-start;
			// wordLength *= wordLength;
			if(!storeOffsets){
				rowWordCounter[row]++;
				atomicAdd(count,1);				
			}else{
				wordSplits[rowWordOffsets[row]+words].start = start;
				wordSplits[rowWordOffsets[row]+words].len 	= wordLength;
				words++;
			}
		}
	}
};




} // hornets_nest namespace
