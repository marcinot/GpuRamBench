#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "assert_cl.h"
#include <time.h> 
#include <string.h>

#define MAX_SOURCE_SIZE (0x100000)

const int LIST_SIZE = 256 * 1024 * 1024;
const int ITERATIONS = 10;
const int KERNEL_ITERATIONS = 2048;
const int BATCH_SIZE = 64*1024;

const char* OPENCL_SOURCE_STR = 
"\n"
"typedef unsigned int uint32_t;\n"
"\n"
"#define LIST_SIZE  256 * 1024 * 1024\n"
"#define MASK (LIST_SIZE -1)\n"
"#define KERNEL_ITERATIONS 2048\n"
"\n"
"/* The state word must be initialized to non-zero */\n"
"uint32_t xorshift32(uint32_t *a)\n"
"{\n"
"\t/* Algorithm \"xor\" from p. 4 of Marsaglia, \"Xorshift RNGs\" */\n"
"\tuint32_t x = *a;\n"
"\tx ^= x << 13;\n"
"\tx ^= x >> 17;\n"
"\tx ^= x << 5;\n"
"\treturn *a = x;\n"
"}\n"
"\n"
"\n"
"\n"
"__kernel void ram_bench(__global const unsigned char *A,  __global unsigned char *C, int seed) {\n"
"\n"
"    // Get the index of the current element to be processed\n"
"    int i = get_global_id(0);\n"
"\t\n"
"\tuint32_t a = i + seed;\n"
"\n"
"\tfor(int j=0; j<KERNEL_ITERATIONS; j++)\n"
"\t{\n"
"\t\tunsigned char v = A[a & MASK];\t\t\n"
"\t\ta = a + v;\n"
"\t\txorshift32(&a);\n"
"\t}\n"
"\n"
"    C[i] = a & 0xff;\n"
"}";



int main(void) {
    printf("started running\n");
    
    int i;
    
    unsigned char* A = (unsigned char*)malloc(LIST_SIZE);    

    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = (i * 1137) % 256;        
    }

    /*
    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("ram_bench.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("kernel loading done\n");
    */


    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;


    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_platform_id* platforms = NULL;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    cl_ok(ret);

    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1,
        &device_id, &ret_num_devices);
    cl_ok(ret);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_ok(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    cl_ok(ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE , NULL, &ret);
    cl_ok(ret);

    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        BATCH_SIZE, NULL, &ret);
    cl_ok(ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE, A, 0, NULL, NULL);
    cl_ok(ret);


    size_t source_size = strlen(OPENCL_SOURCE_STR);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&OPENCL_SOURCE_STR, (const size_t*)&source_size, &ret);
    cl_ok(ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_ok(ret);


    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "ram_bench", &ret);
    cl_ok(ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    cl_ok(ret);

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&c_mem_obj);
    cl_ok(ret);
    
        // Calculate the time taken by fun() 
    clock_t t;
    t = clock();

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        int seed = iter*17;

        ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&seed);
        cl_ok(ret);

        cl_event event;        
        // Execute the OpenCL kernel on the list
        size_t global_item_size = BATCH_SIZE; // Process the entire lists
        size_t local_item_size = 64; // Divide work items into groups of 64
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, &event);
        cl_ok(ret);
        
        clWaitForEvents(1, &event);
        printf("iteration %d / %d\n", iter+1, ITERATIONS);
    }

    t = clock() - t;
    unsigned long int total_bytes = (unsigned long int)ITERATIONS * (unsigned long int)BATCH_SIZE * (unsigned long int)KERNEL_ITERATIONS;

    double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds 

    printf("time_taken = %f \n", time_taken);
    printf("total_bytes = %lu\n", total_bytes);
    printf("memory random byte read speed = %f MB/s\n", (total_bytes / time_taken) / (1024*1024));



    // Read the memory buffer C on the device to the local variable C
    unsigned char* C = (unsigned char*)malloc(LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        BATCH_SIZE, C, 0, NULL, NULL);
    cl_ok(ret);

    unsigned long int sum = 0;
    // Display the result to the screen
    for (i = 0; i < BATCH_SIZE; i++)
        sum += C[i];

    printf("checksum=%lu\n", sum);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);    
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(C);

    system("pause");

    return 0;
}