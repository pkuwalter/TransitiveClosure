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

} // hornets_nest namespace
