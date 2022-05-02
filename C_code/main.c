#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define magic_number_size 4

typedef struct ann {

    int number_of_layers;
    int *layers_sizes;

    double ***w;
    double ***b;

    double ***layers;

    double ***z;

} ANN;

double standard_normal_distribution(int input) {

    double x = (double) input;
    double PI = acos(-1);

    double coefficient = 1 / pow(2 * PI, 0.5);
    double exponent = -pow(x, 2) / 2;
    double result = coefficient * exp(exponent);

    return result;
}


ANN *initialize_ANN(int num_layers, int *layers_sizes) {

    time_t t = time(NULL);
    srand(t);

    ANN *ann = (ANN *) malloc(sizeof(ANN));
    ann->number_of_layers = num_layers;
    ann->layers_sizes = layers_sizes;


    ann->w = (double ***) malloc((num_layers - 1) * sizeof(double **));

    for (int i = 0; i < num_layers - 1; i++) {
        ann->w[i] = (double **) malloc(layers_sizes[i + 1] * sizeof(double *));

        for (int j = 0; j < layers_sizes[i + 1]; j++) {
            ann->w[i][j] = (double *) malloc(layers_sizes[i] * sizeof(double));

            for (int k = 0; k < layers_sizes[i]; k++)
                ann->w[i][j][k] = standard_normal_distribution(rand());
        }
    }


    ann->b = (double ***) malloc((num_layers - 1) * sizeof(double **));

    for (int i = 0; i < num_layers - 1; i++) {
        int neurons = layers_sizes[i + 1];
        ann->b[i] = (double **) malloc(neurons * sizeof(double *));

        for (int j = 0; j < neurons; j++) {
            ann->b[i][j] = (double *) malloc(sizeof(double));
            ann->b[i][j][0] = 0;
        }
    }


    ann->layers = (double ***) malloc(num_layers * sizeof(double **));
    ann->layers[0] = (double **) malloc(layers_sizes[0] * sizeof(double *));

    ann->z = (double ***) malloc((num_layers - 1) * sizeof(double **));

    return ann;
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


double *feed_forward(const double *input, ANN *ann) {

    for (int i = 0; i < ann->layers_sizes[0]; i++)
        ann->layers[0][i][0] = input[i];

    for (int i = 0; i < ann->number_of_layers - 1; i++) {

        double **weights = ann->w[i];
        double **layer = ann->layers[i];
        double **biases = ann->b[i];
        int row = ann->layers_sizes[i + 1];
        int column = ann->layers_sizes[i];

        double **hidden_layer = matrix_multiplication(weights, layer, row, column, 1);
        hidden_layer = matrix_addition(hidden_layer, biases, row, 1);
        ann->z[i] = hidden_layer;
#pragma omp parallel for
        for (int j = 0; j < row; j++)
            hidden_layer[j][0] = sigmoid(hidden_layer[j][0]);
        ann->layers[i + 1] = hidden_layer;
    }

    int output_layer_size = ann->layers_sizes[ann->number_of_layers - 1];
    double *result = (double *) malloc( output_layer_size * sizeof(double));
    for (int i = 0; i < output_layer_size; i++)
        result[i] = ann->layers[ann->number_of_layers - 1][i][0];

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
    int number_of_layers = 4;
    int layers_sizes[] = {784, 16, 16, 10};
    initialize_ANN(number_of_layers, layers_sizes);
    return 0;
}
