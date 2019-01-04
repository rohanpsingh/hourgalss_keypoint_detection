#ifndef GET_PEAKS_H
#define GET_PEAKS_H

#include "common_headers.h"

std::vector<float> getPeakCoordinates(const THFloatTensor* dataTensor, const float* datablock){

    unsigned int nmaps = THFloatTensor_size(dataTensor,0);
    unsigned int side = THFloatTensor_size(dataTensor,1);         //=width=THFloatTensor_size(hmapsTensor,2)
    std::vector<float> peaks;
    for (unsigned int k = 0; k < nmaps; k++){
        double sum_u = 0;
	double sum_v = 0;
	double sum_hm = 0;
        for (unsigned int j = 0; j < side; j++){
	    for (unsigned int i = 0; i < side; i++){
	        double val = *(datablock + k*side*side + j*side + i);
		sum_hm += val;
		sum_u += i*val;
		sum_v += j*val;
	    }
	}
	peaks.push_back(sum_u/sum_hm);
	peaks.push_back(sum_v/sum_hm);
    }
    return peaks;
}

#endif // GET_PEAKS_H
