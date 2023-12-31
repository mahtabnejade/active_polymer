//#include <stdio.h>

/*int reducefile_traj() {
    char inputFileName1[] = "0.1_mpcdtraj.xyz";
    char outputFileName1[] = "0.1_mpcdtraj_reduced.xyz";
    int skipFactor1 = 1000;  // Adjust this to control the level of reduction

    FILE *inputFile1, *outputFile;
    char line[70*60*60*10];  // Assuming a maximum line length of 256 characters

    inputFile1 = fopen(inputFileName1, "r");
    outputFile = fopen(outputFileName1, "w");

    if (inputFile1 == NULL || outputFile == NULL) {
        perror("Error opening files");
        return 1;
    }

    int lineCounter = 0;

    while (fgets(line, sizeof(line), inputFile1) != NULL) {
        if (lineCounter % skipFactor1 == 0) {
            // Write the line to the output file
            fprintf(outputFile, "%s", line);
        }
        lineCounter++;
    }

    fclose(inputFile1);
    fclose(outputFile);

    return 0;
}

int reducefile_vel() {
    char inputFileName2[] = "0.1_mpcdvel.xyz";
    char outputFileName2[] = "0.1_mpcdvel_reduced.xyz";
    int skipFactor2 = 1000;  // Adjust this to control the level of reduction

    FILE *inputFile2, *outputFile2;
    char line2[70*60*60*10];  // Assuming a maximum line length of 256 characters

    inputFile2 = fopen(inputFileName2, "r");
    outputFile2 = fopen(outputFileName2, "w");

    if (inputFile2 == NULL || outputFile2 == NULL) {
        perror("Error opening files");
        return 1;
    }

    int lineCounter2 = 0;

    while (fgets(line2, sizeof(line2), inputFile2) != NULL) {
        if (lineCounter2 % skipFactor2 == 0) {
            // Write the line to the output file
            fprintf(outputFile2, "%s", line2);
        }
        lineCounter2++;
    }

    fclose(inputFile2);
    fclose(outputFile2);

    return 0;
}*/

__global__ void reduceTraj(double *d_x,double *d_y, double *d_z, double *d_xx, double *d_yy, double *d_zz, int N, int skipfactor, double *roundedNumber_x,double *roundedNumber_y,double *roundedNumber_z, int *zerofactorr){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int tidd = int(tid/skipfactor);
    int decimalPlaces = 3; // Number of decimal places to keep
    
 
    if (tid<N)
    {

        if (tid%skipfactor == 0)
        {

           
            //if (d_x[tid] < 5.0 && d_y[tid] < 5.0 && d_z[tid] < 5.0 && d_x[tid] > -5.0 && d_y[tid] > -5.0 && d_z[tid] > -5.0) {

                //printf("*&*\n");

                roundedNumber_x[tid] = roundf(d_x[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_x[tid]=d_x[tid];
                roundedNumber_y[tid] = roundf(d_y[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_y[tid]=d_y[tid];
                roundedNumber_z[tid]= roundf(d_z[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_z[tid]=d_z[tid];

                d_xx[tidd]=roundedNumber_x[tid];
                d_yy[tidd]=roundedNumber_y[tid];
                d_zz[tidd]=roundedNumber_z[tid];
            //}
            //else{
            //    zerofactorr[tid] = 1;
            //    d_xx[tidd]=1000.0000000;
            //    d_yy[tidd]=1000.0000000;
            //    d_zz[tidd]=1000.0000000;
            //}
        }
        
    }
  
}

__host__ void reducetraj(std::string basename, double *d_x,double *d_y, double *d_z,double *d_xx, double *d_yy, double *d_zz, int N, int skipfactor,int grid_size, double *roundedNumber_x,double *roundedNumber_y,double *roundedNumber_z, int *zerofactorr, int *zerofactorrsumblock, int blockSize_ ,int grid_size_){


    int NN = int(N/skipfactor);
    int shared_mem_size_ = 3 * blockSize_ * sizeof(int);
    int block_sum_zerofactorr[grid_size_];

    reduceTraj<<<grid_size, blockSize>>>(d_x, d_y, d_z, d_xx, d_yy, d_zz, N, skipfactor, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr);
    //in this line we should sum over all zerofactorr elements to calculate zerofactorr_sum
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr, zerofactorrsumblock, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(block_sum_zerofactorr, zerofactorrsumblock, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    int d_zerofactorr_sum = 0;
    for (int j = 0; j < grid_size; j++)
        {
            d_zerofactorr_sum += block_sum_zerofactorr[j];
        }
    
    //printf("number of zeros is = %i\n", d_zerofactorr_sum);
    xyz_trj_mpcd(basename + "_mpcdtraj___reduced.xyz", d_xx, d_yy , d_zz, NN, d_zerofactorr_sum);


}



__global__ void reduceVel( double *d_vx,double *d_vy, double *d_vz, double *d_vxx, double *d_vyy, double *d_vzz, double *d_x, double *d_y, double *d_z, int N, int skipfactor, double *roundedNumber_vx,double *roundedNumber_vy,double *roundedNumber_vz, int *zero_factor){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int tidd = int(tid/skipfactor);
    int decimalPlacess = 3; // Number of decimal places to keep
    

    if (tid<N)
    {

        if (tid%skipfactor == 0)
        {
            

            //if (d_x[tid] < 5 && d_y[tid] < 5 && d_z[tid] < 5 && d_x[tid] > -5 && d_y[tid] > -5 && d_z[tid] > -5 ){

                roundedNumber_vx[tid] = roundf(d_vx[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vx[tid]=d_vx[tid];
           
                roundedNumber_vy[tid] = roundf(d_vy[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vy[tid]=d_vy[tid];
            
                roundedNumber_vz[tid]= roundf(d_vz[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vz[tid]=d_vz[tid];

                d_vxx[tidd]=roundedNumber_vx[tid];
                d_vyy[tidd]=roundedNumber_vy[tid];
                d_vzz[tidd]=roundedNumber_vz[tid];
            //}
            //else{
            //    zero_factor[tid] = 1;
                //printf("*");
            //    d_vxx[tidd]=1000.0000000;
            //    d_vyy[tidd]=1000.0000000;
            //    d_vzz[tidd]=1000.0000000;
            //}
        } 
    }

}

__global__ void startend_points(double *d_xx, double *d_yy, double *d_zz, double *d_vxx, double *d_vyy, double *d_vzz, double *endp_x, double *endp_y, double *endp_z, int NN, double *scalefactor){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid<NN){

        endp_x[tid]=d_xx[tid]+d_vxx[tid]* *scalefactor;
        endp_y[tid]=d_yy[tid]+d_vyy[tid]* *scalefactor;
        endp_z[tid]=d_zz[tid]+d_vzz[tid]* *scalefactor;

    }

}
//only for mpcd to reduce the data
__host__ void reducevel(std::string basename, double *d_vx,double *d_vy, double *d_vz,double *d_vxx, double *d_vyy, double *d_vzz, double *d_x, double *d_y, double *d_z, int N, int skipfactor,int grid_size, double *roundedNumber_vx,double *roundedNumber_vy,double *roundedNumber_vz, int *zerofactor, int *zerofactorsumblock, int blockSize_ , int grid_size_){


    int NN = int(N/skipfactor);
    int shared_mem_size_ = 3 * blockSize_ * sizeof(int);
    int block_sum_zerofactor[grid_size_];

    reduceVel<<<grid_size, blockSize>>>(d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, d_x, d_y, d_z, N, skipfactor, roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor);
    //in this line we should sum over all zerofactor elements to calculate zerofactor_sum
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor, zerofactorsumblock, N);
    cudaMemcpy(block_sum_zerofactor, zerofactorsumblock, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    int d_zerofactor_sum = 0;
    for (int j = 0; j < grid_size; j++)
        {
            d_zerofactor_sum += block_sum_zerofactor[j];
           
        }

    
    xyz_trj_mpcd(basename + "_mpcdvel___reduced.xyz", d_vxx, d_vyy , d_vzz, NN, d_zerofactor_sum);

}
 
__host__ void xyz_veltraj_both(std::string basename, double *d_xx, double *d_yy, double *d_zz, double *d_vxx, double *d_vyy, double *d_vzz, int NN, double *endp_x, double *endp_y, double *endp_z, double *scalefactor, int grid_size){

    xyz_trjvel(basename + "_mpcdtrajvel_both.xyz", d_xx, d_yy , d_zz,d_vxx, d_vyy, d_vzz, NN);
    double *scale__factor;
    cudaMalloc((void**)&scale__factor, sizeof(double));
    cudaMemcpy(scale__factor, scalefactor, sizeof(double) , cudaMemcpyHostToDevice);
    startend_points<<<grid_size, blockSize>>>(d_xx, d_yy, d_zz, d_vxx, d_vyy, d_vzz, endp_x, endp_y, endp_z, NN, scale__factor);
    xyz_trjvel(basename + "_startend.xyz", d_xx, d_yy , d_zz,endp_x ,endp_y , endp_z, NN);
    xyz_trj(basename + "_start.xyz", d_xx, d_yy , d_zz, NN);
    xyz_trj(basename + "_end.xyz", endp_x, endp_y , endp_z, NN);
}


