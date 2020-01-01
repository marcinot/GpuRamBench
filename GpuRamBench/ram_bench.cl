
typedef unsigned int uint32_t;

#define LIST_SIZE  256 * 1024 * 1024
#define MASK (LIST_SIZE -1)
#define KERNEL_ITERATIONS 2048

/* The state word must be initialized to non-zero */
uint32_t xorshift32(uint32_t *a)
{
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	uint32_t x = *a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *a = x;
}



__kernel void ram_bench(__global const unsigned char *A,  __global unsigned char *C, int seed) {

    // Get the index of the current element to be processed
    int i = get_global_id(0);
	
	uint32_t a = i + seed;

	for(int j=0; j<KERNEL_ITERATIONS; j++)
	{
		unsigned char v = A[a & MASK];		
		a = a + v;
		xorshift32(&a);
	}

    C[i] = a & 0xff;
}