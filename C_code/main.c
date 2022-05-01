#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define magic_number_size 4

#define input_layer_size 784
#define hidden_layer1_size 16
#define hidden_layer2_size 16
#define output_layer_size 10

// ANN modules
double **w1;
double **w2;
double **w3;
double **b1;
double **b2;
double **b3;

double **input_layer;
double **hidden_layer1;
double **hidden_layer2;
double **output_layer;

double **z2;
double **z3;
double **z4;
//

double standard_normal_distribution(int input) {

    double x = (double) input;
    double PI = acos(-1);

    double coefficient = 1 / pow(2 * PI, 0.5);
    double exponent = -pow(x, 2) / 2;
    double result = coefficient * exp(exponent);

    return result;
}


void initialize_ANN() {

    time_t t = time(NULL);
    srand(t);

    w1 = (double **) malloc(hidden_layer1_size * sizeof(double *));
    for (int i = 0; i < hidden_layer1_size; i++) {
        w1[i] = (double *) malloc(input_layer_size * sizeof(double));
        for (int j = 0; j < input_layer_size; j++)
            w1[i][j] = standard_normal_distribution(rand());
    }

    w2 = (double **) malloc(hidden_layer2_size * sizeof(double *));
    for (int i = 0; i < hidden_layer2_size; i++) {
        w2[i] = (double *) malloc(hidden_layer1_size * sizeof(double));
        for (int j = 0; j < hidden_layer1_size; j++)
            w2[i][j] = standard_normal_distribution(rand());
    }

    w3 = (double **) malloc(output_layer_size * sizeof(double *));
    for (int i = 0; i < output_layer_size; i++) {
        w3[i] = (double *) malloc(hidden_layer2_size * sizeof(double));
        for (int j = 0; j < hidden_layer2_size; j++)
            w3[i][j] = standard_normal_distribution(rand());
    }

    b1 = (double **) malloc(hidden_layer1_size * sizeof(double *));
    for (int i = 0; i < hidden_layer1_size; i++)
        b1[i] = (double *) malloc(sizeof(double));

    b2 = (double **) malloc(hidden_layer2_size * sizeof(double *));
    for (int i = 0; i < hidden_layer2_size; i++)
        b2[i] = (double *) malloc(sizeof(double));

    b3 = (double **) malloc(output_layer_size * sizeof(double *));
    for (int i = 0; i < output_layer_size; i++)
        b3[i] = (double *) malloc(sizeof(double));

    input_layer = (double **) malloc(input_layer_size * sizeof(double *));
    for (int i = 0; i < input_layer_size; i++)
        input_layer[i] = (double *) malloc(sizeof(double));
}


double **matrix_multiplication(double **matrix1, double **matrix2, int row1, int column1, int column2) {

    double **result = (double **) malloc(row1 * sizeof(double *));
    for (int i = 0; i < column2; i++)
        result[i] = (double *) malloc(column2 * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < row1; i++) {

#pragma omp parallel for
        for (int j = 0; j < column2; j++) {
            for (int k = 0; k < column1; k++)
                result[i][j] += matrix1[i][k] * matrix2[k][j];
        }
    }

    return result;
}


double **matrix_addition(double **matrix1, double **matrix2, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] + matrix2[i][j];
    }

    return result;
}


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


double *feed_forward(const double *input) {

    for (int i = 0; i < input_layer_size; i++)
        input_layer[i][0] = input[i];

    hidden_layer1 = matrix_multiplication(w1, input_layer, hidden_layer1_size, input_layer_size, 1);
    hidden_layer1 = matrix_addition(hidden_layer1, b1, hidden_layer1_size, 1);
    z2 = hidden_layer1;
#pragma omp parallel for
    for (int i = 0; i < hidden_layer1_size; i++)
        hidden_layer1[i][0] = sigmoid(hidden_layer1[i][0]);

    hidden_layer2 = matrix_multiplication(w2, hidden_layer1, hidden_layer2_size, hidden_layer1_size, 1);
    hidden_layer2 = matrix_addition(hidden_layer2, b2, hidden_layer2_size, 1);
    z3 = hidden_layer2;
#pragma omp parallel for
    for (int i = 0; i < hidden_layer2_size; i++)
        hidden_layer2[i][0] = sigmoid(hidden_layer2[i][0]);

    output_layer = matrix_multiplication(w3, hidden_layer2, output_layer_size, hidden_layer2_size, 1);
    output_layer = matrix_addition(output_layer, b3, output_layer_size, 1);
    z4 = output_layer;
#pragma omp parallel for
    for (int i = 0; i < output_layer_size; i++)
        output_layer[i][0] = sigmoid(output_layer[i][0]);

    double *result = (double *) malloc(output_layer_size * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < output_layer_size; i++)
        result[i] = output_layer[i][0];

    return result;
}


int number_of_train_images;
int number_of_test_images;
int rows;
int columns;
int *train_labels;
int *test_labels;
int **train_images;
int **test_images;


int readInteger(FILE *fp) {

    unsigned char buffer[4];
    fread(buffer, 1, 4, fp);

    int result = 0;
    for (int i = 0; i < 4; i++) {
        result *= 256;
        result += buffer[i];
    }

    return result;
}


int **readImages(char *address, int t_s) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int images = readInteger(fp);
    rows = readInteger(fp);
    columns = readInteger(fp);
    int pixels = rows * columns;

    int **data = (int **) malloc(images * sizeof(int *));
    for (int i = 0; i < images; i++)
        data[i] = (int *) malloc(pixels * sizeof(int));

    unsigned char pixel;
    for (int i = 0; i < images; i++) {

        for (int j = 0; j < pixels; j++) {

            fread(&pixel, 1, 1, fp);
            data[i][j] = pixel;
        }
    }

    // variable t_s indicates whether this function is reading train data or test data
    if (t_s == 1)
        number_of_train_images = images;
    else
        number_of_test_images = images;

    return data;
}


int *readLabels(char *address) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int items = readInteger(fp);
    int *labels = (int *) malloc(items * sizeof(int));
    unsigned char label;
    for (int i = 0; i < items; i++) {
        fread(&label, 1, 1, fp);
        labels[i] = label;
    }

    return labels;
}


void getDataset() {

    train_images = readImages("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\train-images.idx3-ubyte", 1);
    test_images = readImages("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\t10k-images.idx3-ubyte", 0);
    train_labels = readLabels("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\train-labels.idx1-ubyte");
    test_labels = readLabels("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\t10k-labels.idx1-ubyte");

    for (int i = 0; i < number_of_train_images; i++) {
        for (int j = 0; j < rows * columns; j++)
            train_images[i][j] /= 256;
    }

    for (int i = 0; i < number_of_test_images; i++) {
        for (int j = 0; j < rows * columns; j++)
            test_images[i][j] /= 256;
    }
}


int main() {
    getDataset();
    initialize_ANN();
    return 0;
}
