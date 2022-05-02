//
// Created by Mahdi on 5/2/2022.
//

#include <math.h>
#include <string.h>

#ifndef C_CODE_ACTIVATION_FUNCTIONS_H
#define C_CODE_ACTIVATION_FUNCTIONS_H


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


double Relu(double x) {
    if (x <= 0)
        return 0;
    else
        return x;
}


double call(double x, char *function) {

    if (strcmp(function, "sigmoid") == 0)
        return sigmoid(x);
    else if (strcmp(function, "relu") == 0)
        return Relu(x);

    return tan(acos(-1) / 2);
}

#endif //C_CODE_ACTIVATION_FUNCTIONS_H
