//V2 WITH NO SMALL KERNELS (TIME=6 SEC)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix on host
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory on host
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Fused forward kernel: W1*x + b1 -> ReLU -> W2*hidden + b2 -> softmax
__global__ void fused_forward_kernel(double* W1, double* b1, double* W2, double* b2, 
                                    double* input, double* hidden, double* output, 
                                    double* temp, double* softmax_sum, 
                                    int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Compute hidden layer (W1 * input + b1) and apply ReLU
    if (idx < hidden_size) {
        double sum = 0.0;
        for (int j = 0; j < input_size; j++) {
            sum += W1[idx * input_size + j] * input[j];
        }
        hidden[idx] = (sum + b1[idx] > 0) ? (sum + b1[idx]) : 0.0;
    }

    __syncthreads();

    // Step 2: Compute output layer (W2 * hidden + b2)
    if (idx < output_size) {
        double sum = 0.0;
        for (int j = 0; j < hidden_size; j++) {
            sum += W2[idx * hidden_size + j] * hidden[j];
        }
        output[idx] = sum + b2[idx];
        temp[idx] = exp(output[idx]);
    }

    __syncthreads();

    // Step 3: Compute softmax sum using a simple reduction
    if (idx == 0) {
        *softmax_sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            *softmax_sum += temp[j];
        }
    }

    __syncthreads();

    // Step 4: Normalize output for softmax
    if (idx < output_size) {
        output[idx] = temp[idx] / *softmax_sum;
    }
}

// Fused backward kernel: Compute gradients and update weights/biases
__global__ void fused_backward_kernel(double* W1, double* b1, double* W2, double* b2, 
                                     double* input, double* hidden, double* output, 
                                     double* target, double* d_output, double* d_hidden, 
                                     double* temp, double learning_rate, 
                                     int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Step 1: Compute d_output = output - target
    if (idx < output_size && idy == 0) {
        d_output[idx] = output[idx] - target[idx];
    }

    __syncthreads();

    // Step 2: Backpropagate to hidden layer (W2^T * d_output)
    if (idx < hidden_size && idy == 0) {
        double sum = 0.0;
        for (int i = 0; i < output_size; i++) {
            sum += W2[i * hidden_size + idx] * d_output[i];
        }
        temp[idx] = sum;
        d_hidden[idx] = temp[idx] * (hidden[idx] > 0 ? 1.0 : 0.0);
    }

    __syncthreads();

    // Step 3: Update W2 and b2
    if (idx < hidden_size && idy < output_size) {
        W2[idy * hidden_size + idx] -= learning_rate * d_output[idy] * hidden[idx];
    }
    if (idx < output_size && idy == 0) {
        b2[idx] -= learning_rate * d_output[idx];
    }

    __syncthreads();

    // Step 4: Update W1 and b1
    if (idx < input_size && idy < hidden_size) {
        W1[idy * input_size + idx] -= learning_rate * d_hidden[idy] * input[idx];
    }
    if (idx < hidden_size && idy == 0) {
        b1[idx] -= learning_rate * d_hidden[idx];
    }
}

// Neural network structure
typedef struct {
    double** W1;  // host
    double** W2;  // host
    double* b1;   // host
    double* b2;   // host
    double* d_W1; // device
    double* d_W2; // device
    double* d_b1; // device
    double* d_b2; // device
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Allocate device memory
    cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double));

    // Copy to device
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            h_W1[i * INPUT_SIZE + j] = net->W1[i][j];
    cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    free(h_W1);

    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            h_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];
    cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    free(h_W2);

    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    return net;
}

// Forward pass (GPU-accelerated with fused kernel)
void forward(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output, 
             double* d_temp, double* d_softmax_sum) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    fused_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_W1, net->d_b1, net->d_W2, net->d_b2, 
                                                            d_input, d_hidden, d_output, 
                                                            d_temp, d_softmax_sum, 
                                                            INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaGetLastError();
}

// Backward pass (GPU-accelerated with fused kernel)
void backward(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output, 
              double* d_target, double* d_d_output, double* d_d_hidden, double* d_temp) {
    dim3 blockDim(32, 32);
    dim3 gridDim((INPUT_SIZE + 31) / 32, (HIDDEN_SIZE + 31) / 32);
    fused_backward_kernel<<<gridDim, blockDim>>>(net->d_W1, net->d_b1, net->d_W2, net->d_b2, 
                                                 d_input, d_hidden, d_output, d_target, 
                                                 d_d_output, d_d_hidden, d_temp, 
                                                 LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaGetLastError();
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages, 
           double* d_input, double* d_target, double* d_hidden, double* d_output, 
           double* d_d_output, double* d_d_hidden, double* d_temp, double* d_softmax_sum) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Copy input and target to device
            cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

            // Forward pass (GPU)
            forward(net, d_input, d_hidden, d_output, d_temp, d_softmax_sum);
            cudaDeviceSynchronize();

            // Copy output back for loss and accuracy
            double h_output[OUTPUT_SIZE];
            cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * log(h_output[k] + 1e-8);
            }

            // Compute accuracy
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;

            // Backward pass (GPU)
            backward(net, d_input, d_hidden, d_output, d_target, d_d_output, d_d_hidden, d_temp);
            cudaDeviceSynchronize();
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    // Copy weights back to host
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMemcpy(h_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // Update host weights
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = h_W1[i * INPUT_SIZE + j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = h_W2[i * HIDDEN_SIZE + j];
    free(h_W1);
    free(h_W2);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages, 
              double* d_input, double* d_hidden, double* d_output, 
              double* d_temp, double* d_softmax_sum) {
    int correct = 0;
    double h_output[OUTPUT_SIZE];
    for (int i = 0; i < numImages; i++) {
        cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        forward(net, d_input, d_hidden, d_output, d_temp, d_softmax_sum);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network (No Smaller kernels)\n\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    // Allocate device memory
    double *d_input, *d_target, *d_hidden, *d_output, *d_d_output, *d_d_hidden, *d_temp, *d_softmax_sum;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_temp, OUTPUT_SIZE * sizeof(double)); // For softmax temp
    cudaMalloc(&d_softmax_sum, sizeof(double)); // For softmax sum

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000, d_input, d_target, d_hidden, d_output, 
          d_d_output, d_d_hidden, d_temp, d_softmax_sum);
    evaluate(net, test_images, test_labels, 10000, d_input, d_hidden, d_output, 
             d_temp, d_softmax_sum);

    // Free memory
    freeNetwork(net);
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_d_output);
    cudaFree(d_d_hidden);
    cudaFree(d_temp);
    cudaFree(d_softmax_sum);

    // Free host data
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);

    return 0;
}
