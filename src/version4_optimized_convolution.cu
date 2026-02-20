#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memset, memcpy
#include <cuda_runtime.h>
#include <cublas_v2.h>    // Use cuBLAS V2 API
#include <cuda_profiler_api.h>

// -- Network parameters --
#define INPUT_SIZE          (784)
#define HIDDEN_SIZE         (128)
#define OUTPUT_SIZE         (10)
#define LEARNING_RATE       (0.1f) // Switched to float
#define EPOCHS              (3)
#define NUM_CLASSES         (10)

// -- Batch processing configuration --
#define MAX_BATCH_SIZE      (256)
#define NUM_STREAMS         (4)

// -- Thread block dimensions --
#define BLOCK_SIZE_1D       (256)
// Block sizes for 2D kernels (bias add, activations) - can reuse 1D
#define BLOCK_SIZE_X        (16) // Kept for reference, may not be used directly by cuBLAS replacements
#define BLOCK_SIZE_Y        (16) // Kept for reference

// -- CUDA Error Checking Macros --
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA ERROR in file %s at line %d: %s (%d)\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS ERROR in file %s at line %d: %d\n",      \
                    __FILE__, __LINE__, status);                             \
            /* Note: cuBLAS doesn't have a standard error string function like CUDA */ \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


/* --- Timing Function (CPU) --- */
double measure_cpu_time(clock_t start_time)
{
    return (double)(clock() - start_time) / CLOCKS_PER_SEC;
}

/* --- GPU Timer Utilities --- */
typedef struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

void init_gpu_event_timer(GpuTimer* timer)
{
    CUDA_CHECK(cudaEventCreate(&timer->start));
    CUDA_CHECK(cudaEventCreate(&timer->stop));
    CUDA_CHECK(cudaEventRecord(timer->start, 0));
}

float record_gpu_event_time(GpuTimer* timer)
{
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventRecord(timer->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(timer->stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer->start, timer->stop));
    CUDA_CHECK(cudaEventDestroy(timer->start));
    CUDA_CHECK(cudaEventDestroy(timer->stop));
    return elapsed_ms / 1000.0f; /* Return seconds */
}

/* --- Host Memory Allocation (Now for float) --- */
float* create_host_flt_vector(size_t vector_size)
{
    float* new_vector = (float*)malloc(vector_size * sizeof(float));
    if (new_vector == NULL) {
        perror("Could not allocate host vector memory");
        exit(EXIT_FAILURE);
    }
    return new_vector;
}

float** create_host_ptr_array(int num_rows)
{
    float** matrix_ptrs = (float**)malloc(num_rows * sizeof(float*));
    if (!matrix_ptrs) {
        perror("Could not allocate host matrix row pointers");
        exit(EXIT_FAILURE);
    }
    return matrix_ptrs;
}

void destroy_host_ptr_array(float** matrix_ptrs) // Removed num_rows as it's not needed
{
    if (matrix_ptrs != NULL) {
        free(matrix_ptrs);
    }
}

float** create_host_flt_matrix(int num_rows, int num_cols)
{
    float** new_matrix = (float**)malloc(num_rows * sizeof(float*));
    if (!new_matrix) {
        perror("Could not allocate standard matrix rows");
        exit(EXIT_FAILURE);
    }
    size_t total_elements = (size_t)num_rows * num_cols;
    float* matrix_storage = (float*)malloc(total_elements * sizeof(float));
    if (!matrix_storage) {
        perror("Could not allocate standard matrix storage");
        free(new_matrix);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_rows; ++i) {
        new_matrix[i] = matrix_storage + (size_t)i * num_cols;
    }
    return new_matrix;
}

void destroy_host_flt_matrix(float** matrix)
{
    if (matrix != NULL) {
        if (matrix[0] != NULL) {
            free(matrix[0]);
        }
        free(matrix);
    }
}

/* --- Neural Network Data Structure --- */
typedef struct NeuralNetwork {
    /* Device matrices (flattened, primary storage) - NOW FLOAT */
    float* d_W1; // Shape: hidden_size x input_size (Row-major)
    float* d_W2; // Shape: output_size x hidden_size (Row-major)
    float* d_b1; // Shape: hidden_size
    float* d_b2; // Shape: output_size

    /* Device arrays for batch processing - NOW FLOAT */
    float* d_inputs;              /* Current batch input (batch_size x input_size) */
    float* d_hiddens_pre_relu;    /* Hidden layer pre-activation (batch_size x hidden_size) - NEW */
    float* d_hiddens;             /* Hidden layer activations (post-ReLU) (batch_size x hidden_size) */
    float* d_outputs;             /* Final layer output (post-Softmax) (batch_size x output_size) */
    float* d_pre_softmax_outputs; /* Temp store for pre-softmax values (batch_size x output_size) */
    float* d_targets;             /* Current batch target labels (batch_size x output_size) */

    /* Device arrays for gradients - NOW FLOAT */
    float* d_d_hiddens; /* Gradient w.r.t. hidden layer output (batch_size x hidden_size) */
    float* d_d_outputs; /* Gradient w.r.t. output layer output (batch_size x output_size) */
    float* d_temp_backprop; /* Temporary storage for W2^T * d_d_outputs (batch_size x hidden_size) - NEW */

    /* CUDA streams for asynchronous operations */
    cudaStream_t streams[NUM_STREAMS];

    /* cuBLAS Handle */
    cublasHandle_t cublas_handle;

} NeuralNetwork;

/* ============================================================ */
/* ======================= CUDA Kernels ======================= */
/* ============================================================ */

/* --- Helper Kernels (Replacing parts of original kernels) --- */

// Kernel to add bias vector row-wise to a matrix
// matrix: batch_size x num_features
// bias: num_features
__global__ void add_bias_kernel(float* matrix, const float* bias, int batch_size, int num_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index within the feature vector (col)
    int batch_idx = blockIdx.y;                             // Index for the sample in the batch (row)

    if (feature_idx < num_features && batch_idx < batch_size) {
        int matrix_offset = batch_idx * num_features + feature_idx;
        matrix[matrix_offset] += bias[feature_idx];
    }
}

// Replaces batchReluKernel, operates in-place on input
__global__ void apply_relu_inplace_kernel(float* data_in_out, int size_per_sample, int batch_size)
{
    int element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if ( (element_idx < size_per_sample) && (batch_idx < batch_size) )
    {
        int global_offset = batch_idx * size_per_sample + element_idx;
        float current_value = data_in_out[global_offset];
        data_in_out[global_offset] = fmaxf(0.0f, current_value); // Use fmaxf for float
    }
}


// Renamed from computeOutputErrorKernel
__global__ void calculate_output_delta_kernel(float* d_outputs, float* d_targets, float* d_delta_outputs,
                                      int output_size, int batch_size)
{
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if ( (o_idx < output_size) && (batch_idx < batch_size) )
    {
        int global_offset = batch_idx * output_size + o_idx;
        // Gradient of CrossEntropy+Softmax w.r.t. pre-softmax outputs is (post_softmax_output - target)
        d_delta_outputs[global_offset] = d_outputs[global_offset] - d_targets[global_offset];
    }
}


// Computes delta_hidden = (W2^T * delta_output) .* ReLU_Derivative(hidden_activation_pre_relu)
// backprop_error_in = result of W2^T * delta_output (from cuBLAS) (batch_size x hidden_size)
// hidden_activations_pre_relu = hidden values *before* ReLU was applied (batch_size x hidden_size)
// delta_hidden_out = final delta for hidden layer (batch_size x hidden_size)
__global__ void calculate_hidden_delta_relu_deriv_kernel(const float* backprop_error_in,
                                                 const float* hidden_activations_pre_relu,
                                                 float* delta_hidden_out,
                                                 int hidden_size, int batch_size)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; /* Hidden unit index */
    int batch_idx = blockIdx.y;                        /* Batch sample index */

    if ((h_idx < hidden_size) && (batch_idx < batch_size))
    {
        int global_offset = batch_idx * hidden_size + h_idx;

        // Get the activation *before* ReLU
        float pre_relu_activation = hidden_activations_pre_relu[global_offset];

        // Derivative of ReLU: 1 if input > 0, 0 otherwise
        float relu_derivative = (pre_relu_activation > 0.0f) ? 1.0f : 0.0f;

        // Apply derivative: delta_hidden = backpropagated_error * relu_derivative
        delta_hidden_out[global_offset] = backprop_error_in[global_offset] * relu_derivative;
    }
}


// Replaces updateBiasesKernel, now uses float
__global__ void adjust_all_biases_kernel(float* d_b1, float* d_b2, float* d_d_hiddens, float* d_d_outputs,
                                 float learning_rate, int hidden_size, int output_size,
                                 int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float inv_batch_size = 1.0f / (float)batch_size;

    // Update hidden layer biases b1
    if (idx < hidden_size && batch_size > 0)
    {
        float b1_delta_sum = 0.0f;
        // Sum delta_hidden across the batch for this bias unit
        for (int b = 0; b < batch_size; ++b)
        {
            b1_delta_sum += d_d_hiddens[b * hidden_size + idx];
        }
        // Average and apply update: b = b - lr * avg_delta
        d_b1[idx] -= learning_rate * b1_delta_sum * inv_batch_size;
    }

    // Update output layer biases b2
    if (idx < output_size && batch_size > 0)
    {
        float b2_delta_sum = 0.0f;
        // Sum delta_output across the batch for this bias unit
        for (int b = 0; b < batch_size; ++b)
        {
            b2_delta_sum += d_d_outputs[b * output_size + idx];
        }
        // Average and apply update: b = b - lr * avg_delta
        d_b2[idx] -= learning_rate * b2_delta_sum * inv_batch_size;
    }
}


/* ============================================================ */
/* ==================== Host Functions ======================== */
/* ============================================================ */

/* --- Softmax on Host (now float) --- */
void compute_softmax_on_host(float* vector, int size)
{
    if (size <= 0) { return; }

    float max_value = vector[0];
    for (int i = 1; i < size; ++i) {
        if (vector[i] > max_value) {
            max_value = vector[i];
        }
    }

    float exp_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        vector[i] = expf(vector[i] - max_value); // Use expf for float
        exp_sum += vector[i];
    }

    float epsilon = 1e-9f;
    if (exp_sum < epsilon) { exp_sum = epsilon; }
    float inv_exp_sum = 1.0f / exp_sum;

    for (int i = 0; i < size; ++i) {
        vector[i] *= inv_exp_sum;
    }
}

/* --- Network Initialization (now float) --- */
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

    /* --- Allocate Memory on GPU Device (float) --- */
    CUDA_CHECK(cudaMalloc((void**)&net->d_W1, w1_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_W2, w2_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b1, b1_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b2, b2_elements * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&net->d_inputs, batch_input_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_hiddens_pre_relu, batch_hidden_elements * sizeof(float))); // New buffer
    CUDA_CHECK(cudaMalloc((void**)&net->d_hiddens, batch_hidden_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_outputs, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_pre_softmax_outputs, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_targets, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_d_hiddens, batch_hidden_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_d_outputs, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_temp_backprop, batch_hidden_elements * sizeof(float))); // New buffer

    /* --- Initialize Weights and Biases on Host (float) --- */
    float* h_W1_temp = create_host_flt_vector(w1_elements);
    float* h_W2_temp = create_host_flt_vector(w2_elements);
    float* h_b1_temp = create_host_flt_vector(b1_elements);
    float* h_b2_temp = create_host_flt_vector(b2_elements);

    srand((unsigned int)time(NULL));

    /* Xavier/Glorot Initialization */
    float w1_init_bound = sqrtf(6.0f / (float)(INPUT_SIZE + HIDDEN_SIZE));
    float w2_init_bound = sqrtf(6.0f / (float)(HIDDEN_SIZE + OUTPUT_SIZE));

    for (size_t i = 0; i < w1_elements; ++i) {
        h_W1_temp[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * w1_init_bound;
    }
    for (size_t i = 0; i < w2_elements; ++i) {
        h_W2_temp[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * w2_init_bound;
    }
    memset(h_b1_temp, 0, b1_elements * sizeof(float));
    memset(h_b2_temp, 0, b2_elements * sizeof(float));

    /* --- Copy Initialization Data from Host to Device --- */
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1_temp, w1_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2_temp, w2_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1_temp, b1_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2_temp, b2_elements * sizeof(float), cudaMemcpyHostToDevice));

    /* --- Clean up temporary host arrays --- */
    free(h_W1_temp);
    free(h_W2_temp);
    free(h_b1_temp);
    free(h_b2_temp);

    /* --- Create CUDA Streams --- */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&net->streams[i], cudaStreamNonBlocking));
    }

    /* --- Create cuBLAS Handle --- */
    CUBLAS_CHECK(cublasCreate(&net->cublas_handle));

    /* --- Enable Tensor Core operations (TF32) if available --- */
    // This allows cuBLAS FP32 routines to use Tensor Cores internally on Volta+ GPUs
    // It's generally safe and provides speedup without requiring full FP16 conversion.
    cublasStatus_t tf32_status = cublasSetMathMode(net->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (tf32_status != CUBLAS_STATUS_SUCCESS) {
         fprintf(stderr, "Warning: Could not enable TF32 math mode for cuBLAS (status: %d). Tensor Cores might not be used for FP32.\n", tf32_status);
         // Continue anyway, cuBLAS will use standard FP32 pathways.
    } else {
        //printf("TF32 compute mode enabled for cuBLAS.\n");
    }


    return net;
}

/* --- Forward Pass Batch Processing (Using cuBLAS) --- */
// batch_outputs_final is optional (used for eval/loss calc)
void process_batch_forward_pass(NeuralNetwork* net, float** batch_images, float** batch_outputs_final, int batch_size, int stream_idx)
{
    cudaStream_t current_stream = net->streams[stream_idx];
    CUBLAS_CHECK(cublasSetStream(net->cublas_handle, current_stream)); // Associate cuBLAS with the stream

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /* 1. Prepare and copy input batch to device */
    size_t input_batch_bytes = (size_t)batch_size * INPUT_SIZE * sizeof(float);
    float* h_input_batch_temp = create_host_flt_vector((size_t)batch_size * INPUT_SIZE);
    for (int i = 0; i < batch_size; ++i) {
        memcpy(h_input_batch_temp + (size_t)i * INPUT_SIZE, batch_images[i], INPUT_SIZE * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpyAsync(net->d_inputs, h_input_batch_temp, input_batch_bytes, cudaMemcpyHostToDevice, current_stream));
    free(h_input_batch_temp); // Free host temp buffer immediately after async copy is launched

    /* 2. Layer 1: Matrix Multiplication (Input * W1^T) -> d_hiddens_pre_relu */
    // We want: Hidden_pre = Input(batch x in) * W1^T(in x hidden)
    // cuBLAS expects matrices in column-major, but our data is row-major.
    // Let A=Input, B=W1. To compute C = A * B^T (where A, B, C are conceptually row-major):
    // In cuBLAS: C^T = B * A^T.
    // We can achieve C = A * B^T more directly using: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ...)
    // C(M,N) = A(M,K) * B(K,N)^T where B is stored row-major (N, K)
    // M = batch_size, N = hidden_size, K = input_size
    // A = d_inputs (batch_size x input_size), lda = input_size
    // B = d_W1 (hidden_size x input_size), ldb = input_size
    // C = d_hiddens_pre_relu (batch_size x hidden_size), ldc = hidden_size
    CUBLAS_CHECK(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N, // op(B), op(A) -> W1^T, Input^N
                             HIDDEN_SIZE, batch_size, INPUT_SIZE, // N, M, K
                             &alpha,
                             net->d_W1, INPUT_SIZE,    // B (W1), ldb
                             net->d_inputs, INPUT_SIZE, // A (Input), lda
                             &beta,
                             net->d_hiddens_pre_relu, HIDDEN_SIZE // C (Hidden_pre), ldc
                             ));

    /* 3. Layer 1: Add bias b1 -> d_hiddens_pre_relu (in-place) */
    dim3 gridBias1((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockBias1(BLOCK_SIZE_1D);
    add_bias_kernel<<<gridBias1, blockBias1, 0, current_stream>>>(
        net->d_hiddens_pre_relu, net->d_b1, batch_size, HIDDEN_SIZE
    );

    /* 4. Layer 1: Apply ReLU -> d_hiddens (using d_hiddens_pre_relu as input) */
    // We need d_hiddens_pre_relu for backprop, so ReLU cannot be fully in-place on that buffer.
    // Option 1: Copy d_hiddens_pre_relu to d_hiddens, then apply ReLU in-place on d_hiddens.
    // Option 2: Apply ReLU kernel reading from pre_relu, writing to d_hiddens (not in-place kernel).
    // Let's do Option 1 for simplicity with the existing in-place kernel.
    CUDA_CHECK(cudaMemcpyAsync(net->d_hiddens, net->d_hiddens_pre_relu,
                               (size_t)batch_size * HIDDEN_SIZE * sizeof(float),
                               cudaMemcpyDeviceToDevice, current_stream));
    dim3 gridRelu((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockRelu(BLOCK_SIZE_1D);
    apply_relu_inplace_kernel<<<gridRelu, blockRelu, 0, current_stream>>>(
        net->d_hiddens, HIDDEN_SIZE, batch_size
    );


    /* 5. Layer 2: Matrix Multiplication (Hidden * W2^T) -> d_pre_softmax_outputs */
    // We want: Output_pre = Hidden(batch x hidden) * W2^T(hidden x output)
    // C(M,N) = A(M,K) * B(K,N)^T where B is stored row-major (N, K)
    // M = batch_size, N = output_size, K = hidden_size
    // A = d_hiddens (batch_size x hidden_size), lda = hidden_size
    // B = d_W2 (output_size x hidden_size), ldb = hidden_size
    // C = d_pre_softmax_outputs (batch_size x output_size), ldc = output_size
    CUBLAS_CHECK(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N, // op(B), op(A) -> W2^T, Hidden^N
                             OUTPUT_SIZE, batch_size, HIDDEN_SIZE, // N, M, K
                             &alpha,
                             net->d_W2, HIDDEN_SIZE,    // B (W2), ldb
                             net->d_hiddens, HIDDEN_SIZE, // A (Hidden), lda
                             &beta,
                             net->d_pre_softmax_outputs, OUTPUT_SIZE // C (Output_pre), ldc
                             ));

    /* 6. Layer 2: Add bias b2 -> d_pre_softmax_outputs (in-place) */
    dim3 gridBias2((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockBias2(BLOCK_SIZE_1D);
    add_bias_kernel<<<gridBias2, blockBias2, 0, current_stream>>>(
        net->d_pre_softmax_outputs, net->d_b2, batch_size, OUTPUT_SIZE
    );

    /* --- Perform Softmax on Host --- */
    /* 7. Copy pre-softmax results from Device to Host */
    size_t output_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(float);
    float* h_output_batch_temp = create_host_flt_vector((size_t)batch_size * OUTPUT_SIZE);
    // We need the pre_softmax results for softmax calculation
    CUDA_CHECK(cudaMemcpyAsync(h_output_batch_temp, net->d_pre_softmax_outputs, output_batch_bytes, cudaMemcpyDeviceToHost, current_stream));

    /* 8. Wait for the copy to complete before host processing */
    CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync needed *before* host access

    /* 9. Apply softmax function on the host for each sample */
    for (int i = 0; i < batch_size; ++i) {
        compute_softmax_on_host(h_output_batch_temp + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE);
    }

    /* 10. Copy final post-softmax results from Host back to Device (for backprop loss calc) */
    CUDA_CHECK(cudaMemcpyAsync(net->d_outputs, h_output_batch_temp, output_batch_bytes, cudaMemcpyHostToDevice, current_stream));

    /* 11. If output buffer provided, copy results for host use (evaluation/loss) */
    // This copy happens AFTER softmax on host, so results are correct probabilities
    if (batch_outputs_final != NULL) {
        for (int i = 0; i < batch_size; ++i) {
             memcpy(batch_outputs_final[i], h_output_batch_temp + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(float));
        }
    }

    /* 12. Free temporary host output buffer */
    free(h_output_batch_temp);

    // Kernels in backprop launched on the same stream will wait for the D->H and H->D copies to complete implicitly.
}


/* --- Backward Pass Batch Processing (Using cuBLAS) --- */
void process_batch_backward_pass(NeuralNetwork* net, float** batch_labels, int batch_size, int stream_idx)
{
    cudaStream_t current_stream = net->streams[stream_idx];
    CUBLAS_CHECK(cublasSetStream(net->cublas_handle, current_stream)); // Associate cuBLAS with the stream

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_accumulate = 1.0f; // For accumulating gradients: W = W + beta*dW
    // alpha_update combines learning rate and batch size normalization for weight updates
    const float alpha_update = -LEARNING_RATE / (float)batch_size;

    /* 1. Prepare and copy target labels batch to device */
    size_t target_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(float);
    float* h_target_batch_temp = create_host_flt_vector((size_t)batch_size * OUTPUT_SIZE);
    for (int i = 0; i < batch_size; ++i) {
        memcpy(h_target_batch_temp + (size_t)i * OUTPUT_SIZE, batch_labels[i], OUTPUT_SIZE * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpyAsync(net->d_targets, h_target_batch_temp, target_batch_bytes, cudaMemcpyHostToDevice, current_stream));
    free(h_target_batch_temp); // Free host temp buffer

    /* --- Launch Backward Kernels/Operations (Order Matters!) --- */

    /* 2. Compute Output Layer Error (delta_output = post_softmax_output - target) -> d_d_outputs */
    // This kernel needs the post-softmax outputs (d_outputs) which were copied back from host
    dim3 gridErrOut((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErrOut(BLOCK_SIZE_1D);
    calculate_output_delta_kernel<<<gridErrOut, blockErrOut, 0, current_stream>>>(
        net->d_outputs, net->d_targets, net->d_d_outputs,
        OUTPUT_SIZE, batch_size
    );

    /* 3. Backpropagate Error to Hidden Layer (delta_output * W2) -> d_temp_backprop */
    // We want: Temp = delta_output(batch x output) * W2(output x hidden)
    // C(M,N) = A(M,K) * B(K,N) where A, B, C are row-major
    // In cuBLAS (column-major): C^T = B^T * A^T
    // Simpler: C = A * B using cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ...)
    // M = batch_size, N = hidden_size, K = output_size
    // A = d_d_outputs (batch_size x output_size), lda = output_size
    // B = d_W2 (output_size x hidden_size), ldb = hidden_size
    // C = d_temp_backprop (batch_size x hidden_size), ldc = hidden_size
    CUBLAS_CHECK(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,       // op(B), op(A) -> W2^N, d_d_outputs^N
                             HIDDEN_SIZE, batch_size, OUTPUT_SIZE, // N, M, K
                             &alpha,
                             net->d_W2, HIDDEN_SIZE,         // B (W2), ldb
                             net->d_d_outputs, OUTPUT_SIZE,  // A (d_d_outputs), lda
                             &beta,
                             net->d_temp_backprop, HIDDEN_SIZE // C (Temp), ldc
                             ));

    /* 4. Compute Hidden Layer Delta (apply ReLU derivative) -> d_d_hiddens */
    // delta_hidden = d_temp_backprop * relu_derivative(d_hiddens_pre_relu)
    dim3 gridErrHidden((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErrHidden(BLOCK_SIZE_1D);
    calculate_hidden_delta_relu_deriv_kernel<<<gridErrHidden, blockErrHidden, 0, current_stream>>>(
        net->d_temp_backprop,          // Input: Backpropagated error
        net->d_hiddens_pre_relu,       // Input: Activations *before* ReLU
        net->d_d_hiddens,              // Output: Final hidden delta
        HIDDEN_SIZE, batch_size
    );

    /* 5. Update Weights W2 += alpha_update * (delta_output^T * Hidden) */
    // We want: dW2(output x hidden) = delta_output^T(output x batch) * Hidden(batch x hidden)
    // C(M,N) = A(M,K)^T * B(K,N) where A, B, C are row-major
    // In cuBLAS: C = A^T * B ? No, need C^T = B^T * A
    // Let's use: C = A^T * B
    // M = output_size, N = hidden_size, K = batch_size
    // A = d_d_outputs (batch_size x output_size), lda = output_size
    // B = d_hiddens (batch_size x hidden_size), ldb = hidden_size
    // C = d_W2 (output_size x hidden_size), ldc = hidden_size
    CUBLAS_CHECK(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,       // op(B), op(A) -> Hidden^N, d_d_outputs^T
                             HIDDEN_SIZE, OUTPUT_SIZE, batch_size, // N, M, K
                             &alpha_update, // Includes learning rate and scaling
                             net->d_hiddens, HIDDEN_SIZE,    // B (Hidden), ldb
                             net->d_d_outputs, OUTPUT_SIZE, // A (d_d_outputs), lda
                             &beta_accumulate, // Accumulate: W2 = W2 + update
                             net->d_W2, HIDDEN_SIZE        // C (W2), ldc
                             ));

    /* 6. Update Weights W1 += alpha_update * (delta_hidden^T * Input) */
    // We want: dW1(hidden x input) = delta_hidden^T(hidden x batch) * Input(batch x input)
    // C(M,N) = A(M,K)^T * B(K,N)
    // M = hidden_size, N = input_size, K = batch_size
    // A = d_d_hiddens (batch_size x hidden_size), lda = hidden_size
    // B = d_inputs (batch_size x input_size), ldb = input_size
    // C = d_W1 (hidden_size x input_size), ldc = input_size
     CUBLAS_CHECK(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,       // op(B), op(A) -> Input^N, d_d_hiddens^T
                             INPUT_SIZE, HIDDEN_SIZE, batch_size, // N, M, K
                             &alpha_update, // Includes learning rate and scaling
                             net->d_inputs, INPUT_SIZE,     // B (Input), ldb
                             net->d_d_hiddens, HIDDEN_SIZE,  // A (d_d_hiddens), lda
                             &beta_accumulate, // Accumulate: W1 = W1 + update
                             net->d_W1, INPUT_SIZE         // C (W1), ldc
                             ));

    /* 7. Update Biases b1 and b2 */
    // Kernel sums deltas over batch and applies update
    int max_bias_dim = (HIDDEN_SIZE > OUTPUT_SIZE) ? HIDDEN_SIZE : OUTPUT_SIZE;
    dim3 blockUpdateBias(BLOCK_SIZE_1D);
    dim3 gridUpdateBias((max_bias_dim + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    adjust_all_biases_kernel<<<gridUpdateBias, blockUpdateBias, 0, current_stream>>>(
        net->d_b1, net->d_b2, net->d_d_hiddens, net->d_d_outputs,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // Stream synchronization happens outside this function after the batch loop
}


/* --- Training Function (using float) --- */
void run_network_training(NeuralNetwork* net, float** images, float** labels, int numImages)
{
    clock_t total_cpu_start = clock();
    GpuTimer overall_gpu_timer;
    init_gpu_event_timer(&overall_gpu_timer);

    int effective_batch_size = MAX_BATCH_SIZE;

    /* --- Pre-allocate Host Buffers for Batches (float) --- */
    float** host_batch_outputs = create_host_ptr_array(effective_batch_size);
    float* host_batch_outputs_storage = create_host_flt_vector((size_t)effective_batch_size * OUTPUT_SIZE);
    for (int i = 0; i < effective_batch_size; ++i) {
        host_batch_outputs[i] = host_batch_outputs_storage + (size_t)i * OUTPUT_SIZE;
    }
    float** current_batch_image_ptrs = create_host_ptr_array(effective_batch_size);
    float** current_batch_label_ptrs = create_host_ptr_array(effective_batch_size);

    /* --- Epoch Loop --- */
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        clock_t epoch_cpu_start = clock();
        double current_epoch_loss = 0.0; // Use double for summing loss for precision
        long long epoch_correct_predictions = 0;

        /* Shuffle training data indices */
        int* shuffled_indices = (int*)malloc(numImages * sizeof(int));
        if (!shuffled_indices) { perror("Failed to allocate shuffle indices"); exit(EXIT_FAILURE); }
        for (int i = 0; i < numImages; ++i) { shuffled_indices[i] = i; }
        for (int i = numImages - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int temp_idx = shuffled_indices[i]; shuffled_indices[i] = shuffled_indices[j]; shuffled_indices[j] = temp_idx;
        }

        /* --- Batch Processing Loop --- */
        for (int batch_start_idx = 0; batch_start_idx < numImages; batch_start_idx += effective_batch_size)
        {
            int current_batch_actual_size = fmin(effective_batch_size, numImages - batch_start_idx);
            if (current_batch_actual_size <= 0) { continue; }

            /* Prepare pointers for the current batch */
            for (int i = 0; i < current_batch_actual_size; ++i) {
                int original_idx = shuffled_indices[batch_start_idx + i];
                current_batch_image_ptrs[i] = images[original_idx];
                current_batch_label_ptrs[i] = labels[original_idx];
            }

            /* Select a CUDA stream */
            int stream_index = (batch_start_idx / effective_batch_size) % NUM_STREAMS;

            /* --- Execute Forward Pass --- */
            // This populates host_batch_outputs (already synchronized internally)
            process_batch_forward_pass(net, current_batch_image_ptrs, host_batch_outputs, current_batch_actual_size, stream_index);

            /* --- Calculate Loss & Accuracy (on Host) --- */
            for (int i = 0; i < current_batch_actual_size; ++i) {
                double sample_loss = 0.0;
                int actual_label = -1;
                int predicted_label = 0;

                /* Find actual label and calculate loss */
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    if (current_batch_label_ptrs[i][j] > 0.5f) { // Use float comparison
                        actual_label = j;
                        float predicted_prob = host_batch_outputs[i][j];
                        predicted_prob = fmaxf(predicted_prob, 1e-9f); // Clamp probability (float)
                        sample_loss = -log((double)predicted_prob); // Use log (double) for sum precision
                        break;
                    }
                }
                current_epoch_loss += sample_loss;

                /* Find predicted label */
                float max_prob = host_batch_outputs[i][0];
                for (int j = 1; j < OUTPUT_SIZE; ++j) {
                    if (host_batch_outputs[i][j] > max_prob) {
                        max_prob = host_batch_outputs[i][j];
                        predicted_label = j;
                    }
                }

                if (predicted_label == actual_label) {
                    epoch_correct_predictions++;
                }
            }

            /* --- Execute Backward Pass --- */
            // Launches kernels/cuBLAS calls on the same stream used by forward pass.
            // They will wait for forward pass H->D copies to finish.
            process_batch_backward_pass(net, current_batch_label_ptrs, current_batch_actual_size, stream_index);

        } // End batch loop

        /* --- Synchronize All Streams at Epoch End --- */
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(net->streams[i]));
        }

        /* --- Print Epoch Statistics --- */
        double epoch_avg_loss = current_epoch_loss / (double)numImages;
        double epoch_accuracy = (double)epoch_correct_predictions * 100.0 / (double)numImages;
        double epoch_cpu_time = measure_cpu_time(epoch_cpu_start);
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - CPU Time: %.3fs\n",
               epoch + 1, epoch_avg_loss, epoch_accuracy, epoch_cpu_time);

        free(shuffled_indices);

    } // End epoch loop

    /* --- Free Pre-allocated Host Buffers --- */
    free(host_batch_outputs_storage);
    destroy_host_ptr_array(host_batch_outputs);
    destroy_host_ptr_array(current_batch_image_ptrs);
    destroy_host_ptr_array(current_batch_label_ptrs);

    /* --- Stop Timers and Print Totals --- */
    float total_gpu_time = record_gpu_event_time(&overall_gpu_timer);
    double total_cpu_time = measure_cpu_time(total_cpu_start);
    printf("\nTotal training time: %.3f seconds\n", total_gpu_time);
    
}


/* --- Evaluation Function (using float) --- */
void run_network_evaluation(NeuralNetwork* net, float** images, float** labels, int numImages)
{
    clock_t cpu_eval_start = clock();
    GpuTimer gpu_eval_timer;
    init_gpu_event_timer(&gpu_eval_timer);

    int effective_batch_size = MAX_BATCH_SIZE;
    long long total_correct_predictions = 0;

    /* --- Pre-allocate Host Buffers (float) --- */
    float** host_batch_outputs = create_host_ptr_array(effective_batch_size);
    float* host_batch_outputs_storage = create_host_flt_vector((size_t)effective_batch_size * OUTPUT_SIZE);
    for (int i = 0; i < effective_batch_size; ++i) {
        host_batch_outputs[i] = host_batch_outputs_storage + (size_t)i * OUTPUT_SIZE;
    }
    float** current_batch_image_ptrs = create_host_ptr_array(effective_batch_size);

    /* --- Process Test Data in Batches --- */
    for (int batch_start_idx = 0; batch_start_idx < numImages; batch_start_idx += effective_batch_size)
    {
        int current_batch_actual_size = fmin(effective_batch_size, numImages - batch_start_idx);
        if (current_batch_actual_size <= 0) { continue; }

        /* Prepare pointers for the current batch */
        for (int i = 0; i < current_batch_actual_size; ++i) {
            current_batch_image_ptrs[i] = images[batch_start_idx + i];
        }

        /* --- Execute Forward Pass (use stream 0 for evaluation) --- */
        // Populates host_batch_outputs (already synchronized internally before host access)
        process_batch_forward_pass(net, current_batch_image_ptrs, host_batch_outputs, current_batch_actual_size, 0);

        // No explicit sync needed here IF process_batch_forward_pass syncs before host softmax calculation, which it does.

        /* --- Compare Predictions with Labels (on Host) --- */
        for (int i = 0; i < current_batch_actual_size; ++i) {
            int predicted_label = 0;
            int actual_label = -1;
            int original_idx = batch_start_idx + i;

            /* Find actual label */
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (labels[original_idx][j] > 0.5f) { // float comparison
                    actual_label = j;
                    break;
                }
            }

            /* Find predicted label */
            float max_prob = host_batch_outputs[i][0];
            for (int j = 1; j < OUTPUT_SIZE; ++j) {
                 if (host_batch_outputs[i][j] > max_prob) {
                    max_prob = host_batch_outputs[i][j];
                    predicted_label = j;
                }
            }

            if (predicted_label == actual_label) {
                total_correct_predictions++;
            }
        }
    } // End evaluation batch loop

    /* --- Free Pre-allocated Host Buffers --- */
    free(host_batch_outputs_storage);
    destroy_host_ptr_array(host_batch_outputs);
    destroy_host_ptr_array(current_batch_image_ptrs);

    /* --- Stop Timers and Print Results --- */
    float gpu_eval_time = record_gpu_event_time(&gpu_eval_timer);
    double cpu_eval_time = measure_cpu_time(cpu_eval_start);

    double test_accuracy = (double)total_correct_predictions * 100.0 / (double)numImages;
    printf("Test Accuracy: %.2f%%\n",
           test_accuracy);
}


/* --- MNIST Data Loading Functions (now float) --- */
float** read_mnist_pixel_data(const char* file_path, int num_images)
{
    FILE* image_file = fopen(file_path, "rb");
    if (!image_file) { fprintf(stderr, "ERROR: Cannot open image file %s\n", file_path); exit(EXIT_FAILURE); }
    fseek(image_file, 16, SEEK_SET); // Skip header

    float** image_data = create_host_flt_matrix(num_images, INPUT_SIZE);
    unsigned char* pixel_buffer = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    if (!pixel_buffer) { perror("Failed to allocate pixel buffer"); fclose(image_file); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_images; ++i) {
        if (fread(pixel_buffer, sizeof(unsigned char), INPUT_SIZE, image_file) != INPUT_SIZE) {
            fprintf(stderr, "ERROR: Failed reading image %d from %s\n", i, file_path);
            fclose(image_file); free(pixel_buffer); destroy_host_flt_matrix(image_data); exit(EXIT_FAILURE);
        }
        for (int j = 0; j < INPUT_SIZE; ++j) {
            image_data[i][j] = (float)pixel_buffer[j] / 255.0f; // Normalize to float [0.0, 1.0]
        }
    }
    free(pixel_buffer);
    fclose(image_file);
    return image_data;
}

float** read_mnist_label_data(const char* file_path, int num_labels)
{
    FILE* label_file = fopen(file_path, "rb");
    if (!label_file) { fprintf(stderr, "ERROR: Cannot open label file %s\n", file_path); exit(EXIT_FAILURE); }
    fseek(label_file, 8, SEEK_SET); // Skip header

    float** label_data = create_host_flt_matrix(num_labels, OUTPUT_SIZE);
    unsigned char current_label_byte;

    for (int i = 0; i < num_labels; ++i) {
        if (fread(&current_label_byte, sizeof(unsigned char), 1, label_file) != 1) {
             fprintf(stderr, "ERROR: Failed reading label %d from %s\n", i, file_path);
             fclose(label_file); destroy_host_flt_matrix(label_data); exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            label_data[i][j] = (j == current_label_byte) ? 1.0f : 0.0f; // One-hot encode (float)
        }
    }
    fclose(label_file);
    return label_data;
}


/* --- Resource Cleanup Function --- */
void cleanup_network_resources(NeuralNetwork* net)
{
    if (net == NULL) { return; }

    /* Destroy cuBLAS handle */
    if (net->cublas_handle) {
        CUBLAS_CHECK(cublasDestroy(net->cublas_handle));
    }

    /* Free all allocated device memory */
    if(net->d_W1) CUDA_CHECK(cudaFree(net->d_W1));
    if(net->d_W2) CUDA_CHECK(cudaFree(net->d_W2));
    if(net->d_b1) CUDA_CHECK(cudaFree(net->d_b1));
    if(net->d_b2) CUDA_CHECK(cudaFree(net->d_b2));
    if(net->d_inputs) CUDA_CHECK(cudaFree(net->d_inputs));
    if(net->d_hiddens_pre_relu) CUDA_CHECK(cudaFree(net->d_hiddens_pre_relu));
    if(net->d_hiddens) CUDA_CHECK(cudaFree(net->d_hiddens));
    if(net->d_outputs) CUDA_CHECK(cudaFree(net->d_outputs));
    if(net->d_pre_softmax_outputs) CUDA_CHECK(cudaFree(net->d_pre_softmax_outputs));
    if(net->d_targets) CUDA_CHECK(cudaFree(net->d_targets));
    if(net->d_d_hiddens) CUDA_CHECK(cudaFree(net->d_d_hiddens));
    if(net->d_d_outputs) CUDA_CHECK(cudaFree(net->d_d_outputs));
    if(net->d_temp_backprop) CUDA_CHECK(cudaFree(net->d_temp_backprop));


    /* Destroy CUDA streams */
    for (int i = 0; i < NUM_STREAMS; ++i) {
        if (net->streams[i]) {
            CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
        }
    }

    /* Free the network struct itself */
    free(net);
}

/* ============================================================ */
/* ======================== Main Program ====================== */
/* ============================================================ */

int main(int argc, char* argv[])
{
    printf("MNIST Neural Network (Version 4 with Tensors)\n");

    /* --- Optional: CUDA Device Information --- */
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-enabled devices detected.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
  

    /* --- Load MNIST Dataset (now float) --- */
    const char* train_img_path = "data/train-images.idx3-ubyte";
    const char* train_lbl_path = "data/train-labels.idx1-ubyte";
    const char* test_img_path  = "data/t10k-images.idx3-ubyte";
    const char* test_lbl_path  = "data/t10k-labels.idx1-ubyte";

    // Verify data files exist (simple check)
    FILE *fp_check;
    if ((fp_check = fopen(train_img_path, "rb")) == NULL) { fprintf(stderr,"Error: Cannot find %s. Make sure the 'data' directory exists and contains the MNIST files.\n", train_img_path); return EXIT_FAILURE; } fclose(fp_check);
    if ((fp_check = fopen(train_lbl_path, "rb")) == NULL) { fprintf(stderr,"Error: Cannot find %s.\n", train_lbl_path); return EXIT_FAILURE; } fclose(fp_check);
    if ((fp_check = fopen(test_img_path, "rb")) == NULL) { fprintf(stderr,"Error: Cannot find %s.\n", test_img_path); return EXIT_FAILURE; } fclose(fp_check);
    if ((fp_check = fopen(test_lbl_path, "rb")) == NULL) { fprintf(stderr,"Error: Cannot find %s.\n", test_lbl_path); return EXIT_FAILURE; } fclose(fp_check);


    float** training_images = read_mnist_pixel_data(train_img_path, 60000);
    float** training_labels = read_mnist_label_data(train_lbl_path, 60000);
    float** testing_images  = read_mnist_pixel_data(test_img_path, 10000);
    float** testing_labels  = read_mnist_label_data(test_lbl_path, 10000);
    printf("\n");

    /* --- Initialize Network --- */
    NeuralNetwork* neural_net = build_neural_network_gpu();

    /* --- Train the Network --- */
    run_network_training(neural_net, training_images, training_labels, 60000);

    /* --- Evaluate on Test Set --- */
    run_network_evaluation(neural_net, testing_images, testing_labels, 10000);

    /* --- Clean up Resources --- */
    cleanup_network_resources(neural_net);
    destroy_host_flt_matrix(training_images);
    destroy_host_flt_matrix(training_labels);
    destroy_host_flt_matrix(testing_images);
    destroy_host_flt_matrix(testing_labels);

    /* Optional: Reset CUDA device context */
    // CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
