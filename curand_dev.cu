/*
   Simple usage of curand to generate an amount of numbers on the GPU, it'll use one state per thread, and each thread will create n amount of numbers, sum them
   to create a distribution, and return the sum. Basically, how many of those numbers fit between [0-100[.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
            printf("Error at %s:%d\n",__FILE__,__LINE__); \
            return EXIT_FAILURE;}} while(0)

const char *prog_name;

void usage(FILE *stream, int exit_code)
{
    fprintf(stdout,"Usage: %s options\n",prog_name);
    fprintf(stdout,"\t-h\t\t--help\t\t\tThis help message\n");
    fprintf(stdout,"\t-d type\t\t--distribution type\tDistribution type (valid values: 'uniform', 'normal')\n");
    fprintf(stdout,"\t-s value\t--scale value\t\tScale value for the Normal distribution.\n");
    fprintf(stdout,"\t-S value\t--shift value\t\tShift value for the Normal distribution.\n");
    fprintf(stdout,"\t-m value\t--min value\t\tMinimum value\n");
    fprintf(stdout,"\t-M value\t--max value\t\tMaximum value\n");
    fprintf(stdout,"\t-o file\t\t--output file\t\tOutput CSV file (required)\n");
    fprintf(stdout,"\t-g\t\t--gnuplot\t\tWrite a gnuplot file for showing the output data\n");
    exit(exit_code);
}

__global__ void setup_prng(curandState *state, unsigned long long seed, int amount)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < amount)
        curand_init(seed, tid, 0, &state[tid]);
}

__global__ void create_uniform_distribution(curandState *state, int *distributions, int min, int max, int state_amount, int samples_amount)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < state_amount){
        curandState local_state = state[tid]; // We do this due to performance issue
        int rnd_val = 0;
        for(int i=0;i<samples_amount;i++){
                rnd_val = curand(&local_state) % (max+1 - min) + min;
                if(rnd_val >= min && rnd_val <= max)
                    atomicAdd(&distributions[rnd_val], 1);
        }
        state[tid] = local_state; // Copy back the state
    }
}


__global__ void create_normal_distribution(curandState *state, int *distributions, int min, int max, int state_amount, int samples_amount, int scale, int shift)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
//    int scale = (min-max)*0.1;
//    int shift = (int)(max-min)/2;
    if(tid < state_amount){
        curandState local_state = state[tid]; // We do this due to performance issue
        int rnd_val = 0;
        for(int i=0;i<samples_amount;i++){
            rnd_val = (int)(curand_normal(&local_state) * scale + shift);
            if(rnd_val >= 0 && rnd_val <= (max-min))
                atomicAdd(&distributions[rnd_val], 1);
        }
        state[tid] = local_state; // Copy back the state
    }
}

int main(int argc, char **argv)
{
    prog_name = argv[0];
    const char* const short_options = "ho:d:m:M:gs:S:";
    const struct option long_options[]={
        {"help", 0, NULL, 'h'},
        {"output", 1, NULL, 'o'},
        {"distribution", 1, NULL, 'd'},
        {"min", 1, NULL, 'm'},
        {"max", 1, NULL, 'M'},
        {"gnuplot", 0, NULL, 'g'},
        {"scale", 1, NULL, 's'},
        {"shift", 1, NULL, 'S'},
        {0,0,0,0}
    };
    int next_option;
    int min = 0;
    int max = 100;
    int distribution = 0;
    int plot = 0;
    char *fname = NULL;
    int scale=-1;
    int shift=-1;
    do{
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
        switch(next_option){
            case 'h': usage(stdout,EXIT_SUCCESS); break;
            case 'd': 
                      if(strcmp(optarg, "uniform")==0) distribution=0;
                      else if(strcmp(optarg, "normal")==0) distribution=1;
                      else distribution = 3;                          
                      break;
            case 'o': fname=optarg; break;
            case 'm': min = atoi(optarg); break;
            case 'M': max = atoi(optarg); break;
            case 'g': plot = 1; break;
            case 's': scale = atoi(optarg); break;
            case 'S': shift = atoi(optarg); break;
            case '?': usage(stderr,EXIT_FAILURE); break;
            case -1: break;
            default: abort();
        }
    }while(next_option != -1);
    if(scale<0)
        scale = (min-max)*0.1;
    if(shift<0)
        shift = (max-min)/2;
    int samples_amount = 10000;
    int *h_results, *d_results;
    int qtd_nums = max-min;
    size_t qtd_nums_bytes = sizeof(int)*qtd_nums;
    int blockSize = 512;
    int numThreads = 512;
    curandState *dev_prng_state;
    size_t dev_prng_sate_bytes = sizeof(curandState) * numThreads * blockSize;
    FILE *fp = NULL;
    unsigned long long seed = time(NULL);

    if(fname==NULL)
        usage(stderr,EXIT_FAILURE);

    fprintf(stdout,"[*] Starting %s\n", argv[0]);
    fprintf(stdout,"[*] Parameters:\n");
    fprintf(stdout,"[*]  - Range value: [%d,%d[\n",min,max);
    if(distribution != 3)
        fprintf(stdout,"[*]  - Distribution: %s\n", (distribution == 0) ? "uniform" : "normal");
    else
        fprintf(stdout,"[*] - Distribution: Both\n");
    fprintf(stdout,"[*]  - Scale: %d\n", scale);
    fprintf(stdout,"[*]  - Shift: %d\n", shift);
    fprintf(stdout,"[*]  - Output file: %s\n",fname);
    fprintf(stdout,"[*]  - Plot file: %s\n", (plot==1) ? "True" : "False");
    fprintf(stdout,"[*] Creating file '%s'\n",fname);
    if((fp = fopen(fname, "w"))==NULL){
        perror("fopen");
        return EXIT_FAILURE;
    }

    if(distribution < 3)
        fprintf(fp, "id, qtd\n");
    else
        fprintf(fp, "id, uniform, normal\n");
    fprintf(stdout,"[*] Setting device\n");
    CUDA_CALL(cudaSetDevice(0));

    fprintf(stdout,"[*] Allocating memory\n");
    if((h_results=(int*)malloc(qtd_nums_bytes))==NULL){
        perror("malloc");
        fclose(fp);
        return EXIT_FAILURE;
    }
    memset(h_results,0,qtd_nums_bytes);
    CUDA_CALL(cudaMalloc((void**)&d_results,qtd_nums_bytes));
    CUDA_CALL(cudaMemset(d_results, 0, qtd_nums_bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_prng_state, dev_prng_sate_bytes));;

    fprintf(stdout,"[*] Starting PRNG\n");
    setup_prng<<<blockSize,numThreads>>>(dev_prng_state, seed, 512*512);
    CUDA_CALL(cudaGetLastError());

    fprintf(stdout,"[*] Generating numbers\n");
    if(distribution==3){
        int *tmp = NULL;
        fprintf(stdout,"[*] Allocating temporary array...\n");
        fflush(stdout);
        if((tmp=(int*)malloc(qtd_nums_bytes))==NULL){
            perror("malloc");
            return EXIT_FAILURE;
        }
        memset(tmp, 0, qtd_nums_bytes);
        fprintf(stdout,"[*] Uniform...\n");
        fflush(stdout);
        create_uniform_distribution<<<blockSize, numThreads>>>(dev_prng_state, d_results, min, max, 512*512, samples_amount);
        CUDA_CALL(cudaMemcpy(h_results, d_results, qtd_nums_bytes, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemset(d_results, 0, qtd_nums_bytes));
        fprintf(stdout,"[*] Normal...\n");
        fflush(stdout);
        create_normal_distribution<<<blockSize, numThreads>>>(dev_prng_state, d_results, min, max, 512*512, samples_amount, scale, shift);
        CUDA_CALL(cudaMemcpy(tmp, d_results, qtd_nums_bytes, cudaMemcpyDeviceToHost));
        fprintf(stdout,"[*] Writting on file '%s'\n", fname);
        for(int i=0;i<qtd_nums;i++)
            fprintf(fp, "%d, %d, %d\n", i, h_results[i], tmp[i]);
        free(tmp);
        if(plot == 1){
            fprintf(stdout,"[*] Writting plot commands to file 'multi.plot'\n");
            FILE *fp_plot = NULL;
            if((fp_plot=fopen("multi.plot","w"))==NULL){
                perror("fopen");
                return EXIT_FAILURE;
            }
            fprintf(fp_plot, "set datafile separator \",\"\n");
            fprintf(fp_plot, "set title 'Distribution of %d samples'\n", samples_amount);
            fprintf(fp_plot, "set xlabel 'Bin'\n");
            fprintf(fp_plot, "set ylabel 'Qtd'\n");
            fprintf(fp_plot, "set key outside\n");
            fprintf(fp_plot, "plot '%s' u 1:2 w boxes t 'Uniform', '%s' u 1:3 w boxes t 'Normal'\n", fname, fname);
            fclose(fp_plot);
        }
    }
    else if(distribution == 0)
         create_uniform_distribution<<<blockSize, numThreads>>>(dev_prng_state, d_results, min, max, 512*512, samples_amount);
    else if(distribution == 1)
        create_normal_distribution<<<blockSize, numThreads>>>(dev_prng_state, d_results, min, max, 512*512, samples_amount, scale, shift);
    CUDA_CALL(cudaMemcpy(h_results, d_results, qtd_nums_bytes, cudaMemcpyDeviceToHost));

    if(distribution < 3){
        fprintf(stdout,"[*] Writting on file\n");
        for(int i=0;i<qtd_nums;i++)
            fprintf(fp,"%d, %d\n",i+min, h_results[i]);
    }

    if(plot==1 && distribution < 3){
        fprintf(stdout,"[*] Writting plot commands to file 'rand.plot'\n");
        FILE *fp_plot = NULL;
        if((fp_plot=fopen("rand.plot","w"))==NULL){
            perror("fopen");
            return EXIT_FAILURE;
        }
        fprintf(fp_plot, "set datafile separator \",\"\n");
        fprintf(fp_plot, "set title 'Distribution of %d samples with %s distribution'\n", samples_amount, (distribution==0) ? "Uniform" : "normal");
        fprintf(fp_plot, "set xlabel 'Bin'\n");
        fprintf(fp_plot, "set ylabel 'Qtd'\n");
        fprintf(fp_plot, "plot '%s' u 1:2 w boxes t '%s'\n", fname, (distribution==0) ? "Uniform": "Normal");
        fclose(fp_plot);
    }
    
    fprintf(stdout,"[*] Cleanup\n");
    fclose(fp);
    CUDA_CALL(cudaFree(dev_prng_state));
    CUDA_CALL(cudaFree(d_results));
    free(h_results);

    return EXIT_SUCCESS;
}
