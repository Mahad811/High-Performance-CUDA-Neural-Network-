#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // Kept for potential profiling needs

// -- Network parameters --
#define INPUT_SIZE          (784)
#define HIDDEN_SIZE         (128)
#define OUTPUT_SIZE         (10)
#define LEARNING_RATE       (1.0) // Double precision
#define EPOCHS              (3)
#define NUM_CLASSES         (10)

// -- Batch processing configuration --
#define MAX_BATCH_SIZE      (256)
#define NUM_STREAMS         (4)

// -- Thread block dimensions --
#define BLOCK_SIZE_1D       (256)
#define BLOCK_SIZE_X        (16)
#define BLOCK_SIZE_Y        (16)

// -- CUDA Error Checking Macro --
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA ERROR in file %s at line %d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* --- Timing Function (CPU) --- */
// Renamed from get_time
double measure_cpu_time(clock_t start_time)
{
    return (double)(clock() - start_time) / CLOCKS_PER_SEC;
}

/* --- GPU Timer Utilities --- */
typedef struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

// Renamed from startGpuTimer
void init_gpu_event_timer(GpuTimer* timer)
{
    CUDA_CHECK(cudaEventCreate(&timer->start));
    CUDA_CHECK(cudaEventCreate(&timer->stop));
    CUDA_CHECK(cudaEventRecord(timer->start, 0)); /* Use default stream 0 for event recording */
}

// Renamed from stopGpuTimer
float record_gpu_event_time(GpuTimer* timer)
{
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventRecord(timer->stop, 0));   /* Use default stream 0 */
    CUDA_CHECK(cudaEventSynchronize(timer->stop)); /* Wait for the event to complete */
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer->start, timer->stop));
    CUDA_CHECK(cudaEventDestroy(timer->start));
    CUDA_CHECK(cudaEventDestroy(timer->stop));
    return elapsed_ms / 1000.0f; /* Return seconds */
}

/* --- Host Memory Allocation --- */
// Renamed from allocateHostVector
double* create_host_dbl_vector(size_t vector_size)
{
    double* new_vector = (double*)malloc(vector_size * sizeof(double));
    if (new_vector == NULL)
    {
        perror("Could not allocate host vector memory");
        exit(EXIT_FAILURE);
    }
    return new_vector;
}

// Renamed from allocateHostMatrixPtrs
double** create_host_ptr_array(int num_rows)
{
    double** matrix_ptrs = (double**)malloc(num_rows * sizeof(double*));
    if (!matrix_ptrs)
    {
        perror("Could not allocate host matrix row pointers");
        exit(EXIT_FAILURE);
    }
    return matrix_ptrs;
}

// Renamed from freeHostMatrixPtrs
void destroy_host_ptr_array(double** matrix_ptrs, int num_rows)
{
    /* Assumes underlying row data is managed separately */
    if (matrix_ptrs != NULL)
    {
        free(matrix_ptrs);
    }
}

/* Helper to allocate a standard C-style 2D matrix with contiguous storage */
// Renamed from allocateStandardMatrix
double** create_host_dbl_matrix(int num_rows, int num_cols)
{
    double** new_matrix = (double**)malloc(num_rows * sizeof(double*));
    if (!new_matrix)
    {
        perror("Could not allocate standard matrix rows");
        exit(EXIT_FAILURE);
    }

    size_t total_elements = (size_t)num_rows * num_cols;
    double* matrix_storage = (double*)malloc(total_elements * sizeof(double));
    if (!matrix_storage)
    {
        perror("Could not allocate standard matrix storage");
        free(new_matrix);
        exit(EXIT_FAILURE);
    }

    /* Point row pointers into the single contiguous block */
    for (int i = 0; i < num_rows; ++i)
    {
        new_matrix[i] = matrix_storage + (size_t)i * num_cols;
    }
    return new_matrix;
}

/* Free memory allocated by create_host_dbl_matrix */
// Renamed from freeStandardMatrix
void destroy_host_dbl_matrix(double** matrix)
{
    if (matrix != NULL)
    {
        /* Free the contiguous block using the first row pointer */
        if (matrix[0] != NULL)
        {
            free(matrix[0]);
        }
        /* Free the array of row pointers */
        free(matrix);
    }
}

/* --- Neural Network Data Structure --- */
typedef struct NeuralNetwork {
    /* Device matrices (flattened, primary storage) */
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;

    /* Device arrays for batch processing */
    double* d_inputs;              /* Current batch input */
    double* d_hiddens;             /* Hidden layer activations (post-ReLU) */
    double* d_outputs;             /* Final layer output (post-Softmax) */
    double* d_pre_softmax_outputs; /* Temp store for pre-softmax values */
    double* d_targets;             /* Current batch target labels */

    /* Device arrays for gradients */
    double* d_d_hiddens; /* Gradient w.r.t. hidden layer output */
    double* d_d_outputs; /* Gradient w.r.t. output layer output */

    /* CUDA streams for asynchronous operations */
    cudaStream_t streams[NUM_STREAMS];

} NeuralNetwork;

/* ============================================================ */
/* ======================= CUDA Kernels ======================= */
/* ============================================================ */

/* --- Forward Pass Kernels --- */

// Renamed from forwardLayer1Kernel
__global__ void propagate_to_hidden_kernel(double* W1, double* b1, double* inputs, double* hiddens,
                                  int input_size, int hidden_size, int batch_size)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Hidden unit index */
    int batch_idx = blockIdx.y;                        /* Batch sample index */

    if ( (h_idx < hidden_size) && (batch_idx < batch_size) )
    {
        int input_offset = batch_idx * input_size;   /* Offset for this sample's input */
        int hidden_offset = batch_idx * hidden_size; /* Offset for this sample's hidden state */

        double current_sum = b1[h_idx]; /* Initialize with bias */
        const double* input_vector = inputs + input_offset;
        const double* weight_row = W1 + (size_t)h_idx * input_size; /* Pointer to W1 row */

        /* Compute dot product: W1[h_idx, :] * inputs[batch_idx, :] */
        for (int i = 0; i < input_size; ++i)
        {
            current_sum += weight_row[i] * input_vector[i];
        }

        /* Store result (pre-ReLU activation) */
        hiddens[hidden_offset + h_idx] = current_sum;
    }
}

// Renamed from batchReluKernel
__global__ void apply_relu_batch_kernel(double* data_in_out, int size_per_sample, int batch_size)
{
    int element_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Index within a sample's vector */
    int batch_idx = blockIdx.y;                              /* Index for the sample in the batch */

    if ( (element_idx < size_per_sample) && (batch_idx < batch_size) )
    {
        int global_offset = batch_idx * size_per_sample + element_idx;
        double current_value = data_in_out[global_offset];
        data_in_out[global_offset] = (current_value > 0.0) ? current_value : 0.0; /* In-place ReLU */
    }
}

// Renamed from forwardLayer2Kernel
__global__ void propagate_to_output_kernel(double* W2, double* b2, double* hiddens, double* outputs_pre_softmax,
                                  int hidden_size, int output_size, int batch_size)
{
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Output unit index */
    int batch_idx = blockIdx.y;                        /* Batch sample index */

    if ( (o_idx < output_size) && (batch_idx < batch_size) )
    {
        int hidden_offset = batch_idx * hidden_size;  /* Offset for this sample's hidden state */
        int output_offset = batch_idx * output_size; /* Offset for this sample's output */

        double current_sum = b2[o_idx]; /* Initialize with bias */
        const double* hidden_vector = hiddens + hidden_offset; /* Pointer to hidden activations (post-ReLU) */
        const double* weight_row = W2 + (size_t)o_idx * hidden_size; /* Pointer to W2 row */

        /* Compute dot product: W2[o_idx, :] * hiddens[batch_idx, :] */
        for (int i = 0; i < hidden_size; ++i)
        {
            current_sum += weight_row[i] * hidden_vector[i];
        }

        /* Store result (pre-softmax) */
        outputs_pre_softmax[output_offset + o_idx] = current_sum;
    }
}


/* --- Backward Pass Kernels --- */

// Renamed from computeOutputErrorKernel
__global__ void calculate_output_delta_kernel(double* device_outputs, double* device_targets, double* device_delta_outputs,
                                      int output_size, int batch_size)
{
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Output unit index */
    int batch_idx = blockIdx.y;                        /* Batch sample index */

    if ( (o_idx < output_size) && (batch_idx < batch_size) )
    {
        int global_offset = batch_idx * output_size + o_idx; /* Index within the flattened batch array */
        /* Assumes device_outputs contains POST-softmax values */
        /* Gradient of CrossEntropy+Softmax w.r.t. pre-softmax outputs is (post_softmax_output - target) */
        device_delta_outputs[global_offset] = device_outputs[global_offset] - device_targets[global_offset];
    }
}

// Renamed from computeHiddenErrorKernel
__global__ void calculate_hidden_delta_kernel(double* d_W2, double* d_d_outputs, double* d_hiddens, double* d_d_hiddens,
                                       int hidden_size, int output_size, int batch_size)
{
    /* Computes delta for hidden layer: (W2^T * delta_output) .* ReLU_Derivative(hidden_activation) */
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Hidden unit index */
    int batch_idx = blockIdx.y;                        /* Batch sample index */

    if ( (h_idx < hidden_size) && (batch_idx < batch_size) )
    {
        int hidden_offset = batch_idx * hidden_size;
        int output_offset = batch_idx * output_size;

        double backpropagated_error_sum = 0.0;

        /* Compute W2^T * delta_output for this hidden unit (h_idx) */
        for (int o_idx = 0; o_idx < output_size; ++o_idx)
        {
            /* Access W2 column-wise (transposed): W2[o_idx, h_idx] */
            size_t w2_index = (size_t)o_idx * hidden_size + h_idx;
            /* Access delta_output for this sample: delta_output[batch_idx, o_idx] */
            int d_output_index = output_offset + o_idx;
            backpropagated_error_sum += d_W2[w2_index] * d_d_outputs[d_output_index];
        }

        /* Apply derivative of ReLU activation */
        /* d_hiddens contains the activations *after* ReLU was applied */
        double hidden_activation = d_hiddens[hidden_offset + h_idx];
        double relu_derivative = (hidden_activation > 0.0) ? 1.0 : 0.0;

        /* Store the final delta for the hidden layer */
        d_d_hiddens[hidden_offset + h_idx] = backpropagated_error_sum * relu_derivative;
    }
}


/* --- Weight/Bias Update Kernels --- */

// Renamed from updateWeightsW2Kernel
__global__ void adjust_output_weights_kernel(double* d_W2, double* d_d_outputs, double* d_hiddens,
                                    double learning_rate, int hidden_size, int output_size,
                                    int batch_size)
{
    /* Updates W2 using accumulated gradients over the batch */
    /* W2[o, h] -= LR * Avg(delta_output[:, o] * hiddens[:, h]) */
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Output index (row) */
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y; /* Hidden index (col) */

    if ( (o_idx < output_size) && (h_idx < hidden_size) )
    {
        double gradient_sum = 0.0;

        /* Sum gradient contribution over all samples in the batch */
        for (int b = 0; b < batch_size; ++b)
        {
            int output_offset = b * output_size;
            int hidden_offset = b * hidden_size;
            gradient_sum += d_d_outputs[output_offset + o_idx] * d_hiddens[hidden_offset + h_idx];
        }

        /* Average the gradient and apply the update */
        if (batch_size > 0)
        {
            double average_gradient = gradient_sum / (double)batch_size;
            size_t w2_index = (size_t)o_idx * hidden_size + h_idx;
            d_W2[w2_index] -= learning_rate * average_gradient;
        }
    }
}

// Renamed from updateWeightsW1Kernel
__global__ void adjust_hidden_weights_kernel(double* d_W1, double* d_d_hiddens, double* d_inputs,
                                    double learning_rate, int input_size, int hidden_size,
                                    int batch_size)
{
    /* Updates W1 using accumulated gradients over the batch */
    /* W1[h, i] -= LR * Avg(delta_hidden[:, h] * inputs[:, i]) */
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Hidden index (row) */
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y; /* Input index (col) */

    if ( (h_idx < hidden_size) && (i_idx < input_size) )
    {
        double gradient_sum = 0.0;

        /* Sum gradient contribution over all samples in the batch */
        for (int b = 0; b < batch_size; ++b)
        {
            int hidden_offset = b * hidden_size;
            int input_offset = b * input_size;
            gradient_sum += d_d_hiddens[hidden_offset + h_idx] * d_inputs[input_offset + i_idx];
        }

        /* Average the gradient and apply the update */
        if (batch_size > 0)
        {
            double average_gradient = gradient_sum / (double)batch_size;
            size_t w1_index = (size_t)h_idx * input_size + i_idx;
            d_W1[w1_index] -= learning_rate * average_gradient;
        }
    }
}

// Renamed from updateBiasesKernel
__global__ void adjust_all_biases_kernel(double* d_b1, double* d_b2, double* d_d_hiddens, double* d_d_outputs,
                                 double learning_rate, int hidden_size, int output_size,
                                 int batch_size)
{
    /* Updates biases b1 and b2 based on averaged deltas over the batch */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Update hidden layer biases b1 */
    if (idx < hidden_size)
    {
        double b1_delta_sum = 0.0;
        for (int b = 0; b < batch_size; ++b)
        {
            b1_delta_sum += d_d_hiddens[b * hidden_size + idx];
        }
        if (batch_size > 0)
        {
            double average_delta = b1_delta_sum / (double)batch_size;
            d_b1[idx] -= learning_rate * average_delta;
        }
    }

    /* Update output layer biases b2 */
    /* Note: This relies on blockDim.x being large enough to cover both hidden and output sizes */
    if (idx < output_size)
    {
        double b2_delta_sum = 0.0;
        for (int b = 0; b < batch_size; ++b)
        {
            b2_delta_sum += d_d_outputs[b * output_size + idx];
        }
         if (batch_size > 0)
        {
            double average_delta = b2_delta_sum / (double)batch_size;
            d_b2[idx] -= learning_rate * average_delta;
        }
    }
}

/* ============================================================ */
/* ==================== Host Functions ======================== */
/* ============================================================ */

/* --- Softmax on Host --- */
// Renamed from softmax_host
void compute_softmax_on_host(double* vector, int size)
{
    if (size <= 0) { return; } /* Handle empty vector */

    /* Find maximum value for numerical stability */
    double max_value = vector[0];
    for (int i = 1; i < size; ++i)
    {
        if (vector[i] > max_value)
        {
            max_value = vector[i];
        }
    }

    /* Calculate sum of exponentials */
    double exp_sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        /* Subtract max_value before exponentiating */
        vector[i] = exp(vector[i] - max_value);
        exp_sum += vector[i];
    }

    /* Normalize by the sum */
    double epsilon = 1e-9; /* Prevent division by zero */
    if (exp_sum < epsilon) { exp_sum = epsilon; }

    for (int i = 0; i < size; ++i)
    {
        vector[i] /= exp_sum;
    }
}

/* --- Network Initialization --- */
// Renamed from createNetwork
NeuralNetwork* build_neural_network_gpu()
{
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (net == NULL) {
        perror("Failed to allocate NeuralNetwork struct");
        exit(EXIT_FAILURE);
    }

    /* Calculate sizes */
    size_t w1_elements = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_elements = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t b1_elements = HIDDEN_SIZE;
    size_t b2_elements = OUTPUT_SIZE;
    size_t batch_input_elements = (size_t)MAX_BATCH_SIZE * INPUT_SIZE;
    size_t batch_hidden_elements = (size_t)MAX_BATCH_SIZE * HIDDEN_SIZE;
    size_t batch_output_elements = (size_t)MAX_BATCH_SIZE * OUTPUT_SIZE;

    /* --- Allocate Memory on GPU Device --- */
    CUDA_CHECK(cudaMalloc((void**)&net->d_W1, w1_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_W2, w2_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b1, b1_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b2, b2_elements * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&net->d_inputs, batch_input_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_hiddens, batch_hidden_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_outputs, batch_output_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_pre_softmax_outputs, batch_output_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_targets, batch_output_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_d_hiddens, batch_hidden_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_d_outputs, batch_output_elements * sizeof(double)));

    /* --- Initialize Weights and Biases on Host --- */
    double* h_W1_temp = create_host_dbl_vector(w1_elements);
    double* h_W2_temp = create_host_dbl_vector(w2_elements);
    double* h_b1_temp = create_host_dbl_vector(b1_elements);
    double* h_b2_temp = create_host_dbl_vector(b2_elements);

    srand((unsigned int)time(NULL)); /* Seed random number generator */

    /* Xavier/Glorot Initialization */
    double w1_init_bound = sqrt(6.0 / (double)(INPUT_SIZE + HIDDEN_SIZE));
    double w2_init_bound = sqrt(6.0 / (double)(HIDDEN_SIZE + OUTPUT_SIZE));

    for (size_t i = 0; i < w1_elements; ++i) {
        h_W1_temp[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w1_init_bound;
    }
    for (size_t i = 0; i < w2_elements; ++i) {
        h_W2_temp[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w2_init_bound;
    }
    /* Initialize biases to zero */
    memset(h_b1_temp, 0, b1_elements * sizeof(double));
    memset(h_b2_temp, 0, b2_elements * sizeof(double));

    /* --- Copy Initialization Data from Host to Device --- */
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1_temp, w1_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2_temp, w2_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1_temp, b1_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2_temp, b2_elements * sizeof(double), cudaMemcpyHostToDevice));

    /* --- Clean up temporary host arrays --- */
    free(h_W1_temp);
    free(h_W2_temp);
    free(h_b1_temp);
    free(h_b2_temp);

    /* --- Create CUDA Streams --- */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&net->streams[i], cudaStreamNonBlocking));
    }

    return net;
}

/* --- Forward Pass Batch Processing --- */
// Renamed from forwardBatch
void process_batch_forward_pass(NeuralNetwork* net, double** batch_images, double** batch_outputs_final, int batch_size, int stream_idx)
{
    cudaStream_t current_stream = net->streams[stream_idx];

    /* 1. Prepare and copy input batch to device */
    size_t input_batch_bytes = (size_t)batch_size * INPUT_SIZE * sizeof(double);
    double* h_input_batch_temp = create_host_dbl_vector((size_t)batch_size * INPUT_SIZE);
    for (int i = 0; i < batch_size; ++i) {
        memcpy(h_input_batch_temp + (size_t)i * INPUT_SIZE, batch_images[i], INPUT_SIZE * sizeof(double));
    }
    CUDA_CHECK(cudaMemcpyAsync(net->d_inputs, h_input_batch_temp, input_batch_bytes, cudaMemcpyHostToDevice, current_stream));

    /* 2. Launch Layer 1 Kernel (Input -> Hidden Pre-ReLU) */
    dim3 gridL1((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockL1(BLOCK_SIZE_1D);
    propagate_to_hidden_kernel<<<gridL1, blockL1, 0, current_stream>>>(
        net->d_W1, net->d_b1, net->d_inputs, net->d_hiddens,
        INPUT_SIZE, HIDDEN_SIZE, batch_size
    );

    /* 3. Launch ReLU Activation Kernel (In-place on d_hiddens) */
    dim3 gridRelu((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockRelu(BLOCK_SIZE_1D);
    apply_relu_batch_kernel<<<gridRelu, blockRelu, 0, current_stream>>>(net->d_hiddens, HIDDEN_SIZE, batch_size);

    /* 4. Launch Layer 2 Kernel (Hidden -> Output Pre-Softmax) */
    dim3 gridL2((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockL2(BLOCK_SIZE_1D);
    propagate_to_output_kernel<<<gridL2, blockL2, 0, current_stream>>>(
        net->d_W2, net->d_b2, net->d_hiddens, net->d_pre_softmax_outputs,
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    /* --- Perform Softmax on Host --- */
    /* 5. Copy pre-softmax results from Device to Host */
    size_t output_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(double);
    double* h_output_batch_temp = create_host_dbl_vector((size_t)batch_size * OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(h_output_batch_temp, net->d_pre_softmax_outputs, output_batch_bytes, cudaMemcpyDeviceToHost, current_stream));

    /* 6. Wait for the copy to complete before host processing */
    CUDA_CHECK(cudaStreamSynchronize(current_stream));

    /* 7. Apply softmax function on the host for each sample */
    for (int i = 0; i < batch_size; ++i) {
        compute_softmax_on_host(h_output_batch_temp + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE);
    }

    /* 8. Copy final post-softmax results from Host back to Device (for backprop) */
    CUDA_CHECK(cudaMemcpyAsync(net->d_outputs, h_output_batch_temp, output_batch_bytes, cudaMemcpyHostToDevice, current_stream));

    /* 9. If output buffer provided, copy results for host use (evaluation/loss) */
    if (batch_outputs_final != NULL) {
        for (int i = 0; i < batch_size; ++i) {
             memcpy(batch_outputs_final[i], h_output_batch_temp + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(double));
        }
    }

    /* 10. Free temporary host buffers */
    free(h_input_batch_temp);
    free(h_output_batch_temp);

    /* Note: Backpropagation kernels will be launched on the same stream later. */
    /* The H->D memcpy of d_outputs needs to finish before backprop starts. */
}


/* --- Backward Pass Batch Processing --- */
// Renamed from backwardBatch
void process_batch_backward_pass(NeuralNetwork* net, double** batch_labels, int batch_size, int stream_idx)
{
    cudaStream_t current_stream = net->streams[stream_idx];

    /* 1. Prepare and copy target labels batch to device */
    size_t target_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(double);
    double* h_target_batch_temp = create_host_dbl_vector((size_t)batch_size * OUTPUT_SIZE);
    for (int i = 0; i < batch_size; ++i) {
        memcpy(h_target_batch_temp + (size_t)i * OUTPUT_SIZE, batch_labels[i], OUTPUT_SIZE * sizeof(double));
    }
    CUDA_CHECK(cudaMemcpyAsync(net->d_targets, h_target_batch_temp, target_batch_bytes, cudaMemcpyHostToDevice, current_stream));

    /* --- Launch Backward Kernels (Order Matters!) --- */

    /* 2. Compute Output Layer Error (delta_output = post_softmax_output - target) */
    dim3 gridErrOut((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErrOut(BLOCK_SIZE_1D);
    calculate_output_delta_kernel<<<gridErrOut, blockErrOut, 0, current_stream>>>(
        net->d_outputs, net->d_targets, net->d_d_outputs,
        OUTPUT_SIZE, batch_size
    );

    /* 3. Compute Hidden Layer Error (delta_hidden = (W2^T * delta_output) .* relu_deriv ) */
    dim3 gridErrHidden((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErrHidden(BLOCK_SIZE_1D);
    calculate_hidden_delta_kernel<<<gridErrHidden, blockErrHidden, 0, current_stream>>>(
        net->d_W2, net->d_d_outputs, net->d_hiddens, net->d_d_hiddens,
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    /* 4. Update Weights W2 */
    dim3 blockUpdateW2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridUpdateW2((OUTPUT_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                      (HIDDEN_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    adjust_output_weights_kernel<<<gridUpdateW2, blockUpdateW2, 0, current_stream>>>(
        net->d_W2, net->d_d_outputs, net->d_hiddens,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    /* 5. Update Weights W1 (Requires d_inputs from the forward pass) */
    dim3 blockUpdateW1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridUpdateW1((HIDDEN_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                      (INPUT_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    adjust_hidden_weights_kernel<<<gridUpdateW1, blockUpdateW1, 0, current_stream>>>(
        net->d_W1, net->d_d_hiddens, net->d_inputs,
        LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, batch_size
    );

    /* 6. Update Biases b1 and b2 */
    /* Ensure block size is large enough for max(HIDDEN_SIZE, OUTPUT_SIZE) */
    int max_bias_dim = (HIDDEN_SIZE > OUTPUT_SIZE) ? HIDDEN_SIZE : OUTPUT_SIZE;
    dim3 blockUpdateBias(BLOCK_SIZE_1D);
    dim3 gridUpdateBias((max_bias_dim + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    adjust_all_biases_kernel<<<gridUpdateBias, blockUpdateBias, 0, current_stream>>>(
        net->d_b1, net->d_b2, net->d_d_hiddens, net->d_d_outputs,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    /* 7. Free temporary host target buffer */
    free(h_target_batch_temp);

    /* Note: Stream synchronization happens outside this function after the batch loop */
}


/* --- Training Function --- */
// Renamed from train
void run_network_training(NeuralNetwork* net, double** images, double** labels, int numImages)
{
    clock_t total_cpu_start = clock();
    GpuTimer overall_gpu_timer;
    init_gpu_event_timer(&overall_gpu_timer); /* Start overall GPU timer */

    int effective_batch_size = MAX_BATCH_SIZE;

    /* --- Pre-allocate Host Buffers for Batches --- */
    /* Buffer for receiving batch outputs from forward pass (for loss/accuracy) */
    double** host_batch_outputs = create_host_ptr_array(effective_batch_size);
    double* host_batch_outputs_storage = create_host_dbl_vector((size_t)effective_batch_size * OUTPUT_SIZE);
    for (int i = 0; i < effective_batch_size; ++i) {
        host_batch_outputs[i] = host_batch_outputs_storage + (size_t)i * OUTPUT_SIZE;
    }
    /* Buffers for holding pointers to current batch images/labels */
    double** current_batch_image_ptrs = create_host_ptr_array(effective_batch_size);
    double** current_batch_label_ptrs = create_host_ptr_array(effective_batch_size);

    /* --- Epoch Loop --- */
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        clock_t epoch_cpu_start = clock();
        double current_epoch_loss = 0.0;
        long long epoch_correct_predictions = 0;

        /* Shuffle training data indices for stochasticity */
        int* shuffled_indices = (int*)malloc(numImages * sizeof(int));
        if (!shuffled_indices) { perror("Failed to allocate shuffle indices"); exit(EXIT_FAILURE); }
        for (int i = 0; i < numImages; ++i) { shuffled_indices[i] = i; }
        for (int i = numImages - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int temp_idx = shuffled_indices[i];
            shuffled_indices[i] = shuffled_indices[j];
            shuffled_indices[j] = temp_idx;
        }

        /* --- Batch Processing Loop --- */
        for (int batch_start_idx = 0; batch_start_idx < numImages; batch_start_idx += effective_batch_size)
        {
            int current_batch_actual_size = fmin(effective_batch_size, numImages - batch_start_idx);
            if (current_batch_actual_size <= 0) { continue; } /* Skip if batch is empty */

            /* Prepare pointers to the data for the current batch */
            for (int i = 0; i < current_batch_actual_size; ++i) {
                int original_idx = shuffled_indices[batch_start_idx + i];
                current_batch_image_ptrs[i] = images[original_idx];
                current_batch_label_ptrs[i] = labels[original_idx];
            }

            /* Select a CUDA stream for this batch */
            int stream_index = (batch_start_idx / effective_batch_size) % NUM_STREAMS;

            /* --- Execute Forward Pass --- */
            /* This populates host_batch_outputs with results after host softmax */
            process_batch_forward_pass(net, current_batch_image_ptrs, host_batch_outputs, current_batch_actual_size, stream_index);

            /* --- Calculate Loss & Accuracy (on Host) --- */
            /* process_batch_forward_pass synchronized its stream internally before returning results */
            for (int i = 0; i < current_batch_actual_size; ++i) {
                double sample_loss = 0.0;
                int actual_label = -1;
                int predicted_label = 0; /* Assume label 0 initially */

                /* Find actual label and calculate loss component */
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    if (current_batch_label_ptrs[i][j] > 0.5) { /* Check for the '1' in one-hot */
                        actual_label = j;
                        double predicted_prob = host_batch_outputs[i][j];
                        /* Clamp probability to avoid log(0) */
                        predicted_prob = (predicted_prob < 1e-9) ? 1e-9 : predicted_prob;
                        sample_loss = -log(predicted_prob);
                        break; /* Found the actual label */
                    }
                }
                current_epoch_loss += sample_loss;

                /* Find predicted label (index with highest probability) */
                for (int j = 1; j < OUTPUT_SIZE; ++j) {
                    if (host_batch_outputs[i][j] > host_batch_outputs[i][predicted_label]) {
                        predicted_label = j;
                    }
                }

                /* Check if prediction is correct */
                if (predicted_label == actual_label) {
                    epoch_correct_predictions++;
                }
            } // End loop over samples in batch

            /* --- Execute Backward Pass --- */
            /* Launches kernels on the same stream used by forward pass */
            /* Kernels will implicitly wait for the H->D copy of d_outputs to finish */
            process_batch_backward_pass(net, current_batch_label_ptrs, current_batch_actual_size, stream_index);

            /* --- Optional: Progress Indicator --- */
            int batches_processed = (batch_start_idx / effective_batch_size) + 1;
            if (batches_processed % 50 == 0) {
                fflush(stdout);
            }

        } // End batch loop for the epoch

        /* --- Synchronize All Streams at Epoch End --- */
        /* Ensures all weight updates from the epoch are complete */
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(net->streams[i]));
        }

        /* --- Print Epoch Statistics --- */
        double epoch_avg_loss = current_epoch_loss / (double)numImages;
        double epoch_accuracy = (double)epoch_correct_predictions * 100.0 / (double)numImages;
        double epoch_cpu_time = measure_cpu_time(epoch_cpu_start);
	printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1,  epoch_avg_loss, epoch_accuracy, epoch_cpu_time);
 
        free(shuffled_indices); /* Free indices for this epoch */

    } // End epoch loop

    /* --- Free Pre-allocated Host Buffers --- */
    free(host_batch_outputs_storage);
    destroy_host_ptr_array(host_batch_outputs, effective_batch_size);
    destroy_host_ptr_array(current_batch_image_ptrs, effective_batch_size);
    destroy_host_ptr_array(current_batch_label_ptrs, effective_batch_size);

    /* --- Stop Timers and Print Totals --- */
    float total_gpu_time = record_gpu_event_time(&overall_gpu_timer);
    double total_cpu_time = measure_cpu_time(total_cpu_start);
    printf("\nTotal training time: %.3f seconds\n", total_gpu_time);
}


/* --- Evaluation Function --- */
// Renamed from evaluate
void run_network_evaluation(NeuralNetwork* net, double** images, double** labels, int numImages)
{
    clock_t cpu_eval_start = clock();
    GpuTimer gpu_eval_timer;
    init_gpu_event_timer(&gpu_eval_timer); /* Start evaluation GPU timer */

    int effective_batch_size = MAX_BATCH_SIZE;
    long long total_correct_predictions = 0;

    /* --- Pre-allocate Host Buffers --- */
    double** host_batch_outputs = create_host_ptr_array(effective_batch_size);
    double* host_batch_outputs_storage = create_host_dbl_vector((size_t)effective_batch_size * OUTPUT_SIZE);
    for (int i = 0; i < effective_batch_size; ++i) {
        host_batch_outputs[i] = host_batch_outputs_storage + (size_t)i * OUTPUT_SIZE;
    }
    double** current_batch_image_ptrs = create_host_ptr_array(effective_batch_size);

    /* --- Process Test Data in Batches --- */
    for (int batch_start_idx = 0; batch_start_idx < numImages; batch_start_idx += effective_batch_size)
    {
        int current_batch_actual_size = fmin(effective_batch_size, numImages - batch_start_idx);
        if (current_batch_actual_size <= 0) { continue; }

        /* Prepare pointers to image data for the current batch */
        for (int i = 0; i < current_batch_actual_size; ++i) {
            current_batch_image_ptrs[i] = images[batch_start_idx + i];
        }

        /* --- Execute Forward Pass (using stream 0 for evaluation) --- */
        /* Populates host_batch_outputs */
        process_batch_forward_pass(net, current_batch_image_ptrs, host_batch_outputs, current_batch_actual_size, 0);

        /* --- Ensure Forward Pass is Complete --- */
        /* Wait for stream 0 to finish */
        CUDA_CHECK(cudaStreamSynchronize(net->streams[0]));

        /* --- Compare Predictions with Labels (on Host) --- */
        for (int i = 0; i < current_batch_actual_size; ++i) {
            int predicted_label = 0;
            int actual_label = -1;
            int original_idx = batch_start_idx + i; /* Index in the full test set */

            /* Find actual label */
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (labels[original_idx][j] > 0.5) {
                    actual_label = j;
                    break;
                }
            }

            /* Find predicted label */
            for (int j = 1; j < OUTPUT_SIZE; ++j) {
                if (host_batch_outputs[i][j] > host_batch_outputs[i][predicted_label]) {
                    predicted_label = j;
                }
            }

            /* Increment count if prediction is correct */
            if (predicted_label == actual_label) {
                total_correct_predictions++;
            }
        }
    } // End batch loop for evaluation

    /* --- Free Pre-allocated Host Buffers --- */
    free(host_batch_outputs_storage);
    destroy_host_ptr_array(host_batch_outputs, effective_batch_size);
    destroy_host_ptr_array(current_batch_image_ptrs, effective_batch_size);

    /* --- Stop Timers and Print Results --- */
    float gpu_eval_time = record_gpu_event_time(&gpu_eval_timer);
    double cpu_eval_time = measure_cpu_time(cpu_eval_start);

    double test_accuracy = (double)total_correct_predictions * 100.0 / (double)numImages;
    printf("Test Accuracy: %.2f%%)\n",
           test_accuracy);
}


/* --- MNIST Data Loading Functions --- */
// Renamed from loadMNISTImages
double** read_mnist_pixel_data(const char* file_path, int num_images)
{
    FILE* image_file = fopen(file_path, "rb");
    if (!image_file) {
        fprintf(stderr, "ERROR: Cannot open image file %s\n", file_path);
        exit(EXIT_FAILURE);
    }
    /* Skip MNIST header (magic number, num images, rows, cols) */
    fseek(image_file, 16, SEEK_SET);

    /* Allocate matrix to store images */
    double** image_data = create_host_dbl_matrix(num_images, INPUT_SIZE);

    /* Buffer for reading one image at a time */
    unsigned char* pixel_buffer = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    if (!pixel_buffer) {
        perror("Failed to allocate pixel buffer");
        fclose(image_file); exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_images; ++i) {
        size_t read_count = fread(pixel_buffer, sizeof(unsigned char), INPUT_SIZE, image_file);
        if (read_count != INPUT_SIZE) {
            fprintf(stderr, "ERROR: Failed reading image index %d from %s\n", i, file_path);
            fclose(image_file); free(pixel_buffer); destroy_host_dbl_matrix(image_data); exit(EXIT_FAILURE);
        }
        /* Normalize pixel values [0, 255] -> [0.0, 1.0] */
        for (int j = 0; j < INPUT_SIZE; ++j) {
            image_data[i][j] = (double)pixel_buffer[j] / 255.0;
        }
        /* Optional: Progress indicator */
        // if ((i + 1) % 10000 == 0) { printf("  ... loaded %d images\n", i + 1); }
    }
    free(pixel_buffer);
    fclose(image_file);
    return image_data;
}

// Renamed from loadMNISTLabels
double** read_mnist_label_data(const char* file_path, int num_labels)
{
    FILE* label_file = fopen(file_path, "rb");
    if (!label_file) {
        fprintf(stderr, "ERROR: Cannot open label file %s\n", file_path);
        exit(EXIT_FAILURE);
    }
    /* Skip MNIST header (magic number, num labels) */
    fseek(label_file, 8, SEEK_SET);

    /* Allocate matrix for one-hot encoded labels */
    double** label_data = create_host_dbl_matrix(num_labels, OUTPUT_SIZE);
    unsigned char current_label_byte; /* Buffer for reading single label */

    for (int i = 0; i < num_labels; ++i) {
        size_t read_count = fread(&current_label_byte, sizeof(unsigned char), 1, label_file); // Use & here
        if (read_count != 1) {
            fprintf(stderr, "ERROR: Failed reading label index %d from %s\n", i, file_path);
            fclose(label_file); destroy_host_dbl_matrix(label_data); exit(EXIT_FAILURE);
        }
        /* Perform one-hot encoding */
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            label_data[i][j] = (j == current_label_byte) ? 1.0 : 0.0;
        }
         /* Optional: Progress indicator */
        // if ((i + 1) % 10000 == 0) { printf("  ... loaded %d labels\n", i + 1); }
    }
    
    fclose(label_file);
    return label_data;
}


/* --- Resource Cleanup Function --- */
// Renamed from freeNetwork
void cleanup_network_resources(NeuralNetwork* net)
{
    if (net == NULL) { return; } /* Nothing to free */


    /* Free all allocated device memory */
    CUDA_CHECK(cudaFree(net->d_W1));
    CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1));
    CUDA_CHECK(cudaFree(net->d_b2));
    CUDA_CHECK(cudaFree(net->d_inputs));
    CUDA_CHECK(cudaFree(net->d_hiddens));
    CUDA_CHECK(cudaFree(net->d_outputs));
    CUDA_CHECK(cudaFree(net->d_pre_softmax_outputs));
    CUDA_CHECK(cudaFree(net->d_targets));
    CUDA_CHECK(cudaFree(net->d_d_hiddens));
    CUDA_CHECK(cudaFree(net->d_d_outputs));

    /* Destroy CUDA streams */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
    }

    /* Free the network struct itself */
    free(net);
}

/* ============================================================ */
/* ======================== Main Program ====================== */
/* ============================================================ */

int main(int argc, char* argv[])
{
    printf("MNIST Neural Network (Version 3)\n\n");

    /* --- Optional: CUDA Device Information --- */
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-enabled devices detected.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0)); /* Use device 0 */
    printf("\n");
    /* --- End Optional Device Info --- */

    /* --- Load MNIST Dataset --- */
    /* Adjust paths if data directory is different */
    const char* train_img_path = "data/train-images.idx3-ubyte";
    const char* train_lbl_path = "data/train-labels.idx1-ubyte";
    const char* test_img_path  = "data/t10k-images.idx3-ubyte";
    const char* test_lbl_path  = "data/t10k-labels.idx1-ubyte";

    double** training_images = read_mnist_pixel_data(train_img_path, 60000);
    double** training_labels = read_mnist_label_data(train_lbl_path, 60000);
    double** testing_images  = read_mnist_pixel_data(test_img_path, 10000);
    double** testing_labels  = read_mnist_label_data(test_lbl_path, 10000);
    
    /* --- Initialize Network --- */
    NeuralNetwork* neural_net = build_neural_network_gpu();
    
    /* --- Train the Network --- */
    run_network_training(neural_net, training_images, training_labels, 60000);

    /* --- Evaluate on Test Set --- */
    run_network_evaluation(neural_net, testing_images, testing_labels, 10000);

    /* --- Clean up Resources --- */
    cleanup_network_resources(neural_net);
    destroy_host_dbl_matrix(training_images);
    destroy_host_dbl_matrix(training_labels);
    destroy_host_dbl_matrix(testing_images);
    destroy_host_dbl_matrix(testing_labels);
    
    /* Optional: Reset CUDA device context */
    // CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
