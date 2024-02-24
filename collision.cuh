
//cell sort, calculating the index of each particle there is a unique ID for each cell  based on their position
//Purpose: This kernel calculates the unique ID for each particle's cell based on their position.
//x,y,z are positions of the particles //L=dimentions of the cells//N=number of particles
//index= Array to store the calculated unique ID (index) for each particle's cell
__global__ void cellSort(double* x,double* y,double* z, double *L, int* index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
    index[tid] = int(x[tid] + L[0] / 2) + L[0] * int(y[tid] + L[1] / 2) + L[0] * L[1] * int(z[tid] + L[2] / 2);
    }

} //Output: The index array will be updated with the computed unique IDs.


//Purpose: This kernel initializes cell-related arrays to prepare for calculations.
//ux,uy,uz are cell velocities.//n is th number of particles in each cell//e is cell energy//Nc is number of cells.
__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n,int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0;
        uy[tid] = 0;
        uz[tid] = 0;
        n[tid] = 0;
        e[tid]=0;
    } 
} //Output: The arrays ux, uy, uz, e, and n will be set to 0.

//Purpose: This kernel calculates the mean velocity and mass of particles in each cell.
//vx, vy, vz: Arrays containing the particle velocities 
__global__ void MeanVelCell(double* ux, double* vx,double* uy, double* vy,double* uz, double* vz,int* index, int *n,int *m, int mass, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        double tmp =vx[tid] *mass;
        atomicAdd(&ux[idxx] , tmp );
        tmp =vy[tid] *mass;
        atomicAdd(&uy[idxx] , tmp );
        tmp =vz[tid] *mass;
        atomicAdd(&uz[idxx] , tmp );
        atomicAdd(&n[idxx] , 1 );
        atomicAdd(&m[idxx], mass);
    }
}  //Output: The ux, uy, uz, n, and m arrays will be updated with the calculated mean velocities and masses for each cell.

__global__ void RotationStep1(double *ux , double *uy ,double *uz,double *rot, int *m ,double *phi , double *theta, int Nc)
//This kernel performs a rotation transformation on cell velocities and calculates rotation matrices for each cell.
//ux, uy, uz: Arrays containing the cell velocities //rot: Array to store the rotation matrices for each cell.
//m: Array containing the mass of particles in each cell. //phi, theta: Arrays containing rotation angles (phi, theta) for each cell.
//Nc: Number of cells.
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha = 13.0 / 18.0 * M_PI;
    double co = cos(alpha), si = sin(alpha);
    if (tid<Nc)
    {
        theta[tid] = theta[tid]* 2 -1; //This line modifies the value of theta for the current particle or cell. 
                                       //It scales the value by 2 and subtracts 1, 
                                       //effectively mapping the value from the range [0, 1] to the range [-1, 1].
        phi[tid] = phi[tid]* M_PI*2;   // It scales the value by 2 * pi (where M_PI is the constant for pi) to map it from the range [0, 1] to the range [0, 2*pi].
        ux[tid] = ux[tid]/m[tid];
        uy[tid] = uy[tid]/m[tid];
        uz[tid] = uz[tid]/m[tid];

        //The next three lines calculate three components (n1, n2, and n3) of a unit vector n based on theta and phi.
        //This unit vector n will be used to construct the rotation matrix in the subsequent lines.
        double n1 = std::sqrt(1 - theta[tid] * theta[tid]) * cos(phi[tid]);
        double n2 = std::sqrt(1 - theta[tid] * theta[tid]) * sin(phi[tid]);
        double n3 = theta[tid];
        
        //The following nine lines calculate the elements of the 3x3 rotation matrix rot for the current particle or cell using the unit vector n and the constants co and si. 
        //The rotation matrix will be stored in the rot array at the appropriate index (tid*9 + i)
        rot[tid*9+0] =n1 * n1 + (1 - n1 * n1) * co ;
        rot[tid*9+1] =n1 * n2 * (1 - co) - n3 * si;
        rot[tid*9+2] =n1 * n3 * (1 - co) + n2 * si;
        rot[tid*9+3] =n1 * n2 * (1 - co) + n3 * si;
        rot[tid*9+4] =n2 * n2 + (1 - n2 * n2) * co;
        rot[tid*9+5] =n2 * n3 * (1 - co) - n1 * si;
        rot[tid*9+6] =n1 * n3 * (1 - co) - n2 * si;
        rot[tid*9+7] =n2 * n3 * (1 - co) + n1 * si;
        rot[tid*9+8] =n3 * n3 + (1 - n3 * n3) * co;
        
    }
} //Output: The rot array will be updated with the calculated rotation matrices.

__global__ void RotationStep2(double *rvx , double *rvy, double *rvz , double *rot , int *index,int N)
//This kernel applies the rotation matrices calculated in the previous step to the particle velocities
//rvx, rvy, rvz: Arrays containing the relative velocities of particles (with respect to their cells).
//rot: Array containing the rotation matrices for each cell
//index: Array containing the unique IDs of each particle's cell.
//N: Number of particles.
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(tid<N)
    {
        const unsigned int idxx = index[tid];//This line retrieves the unique ID (idxx) of the cell associated with the current particle. The index array maps each particle to its corresponding cell using unique IDs.

        double RV[3] = {rvx[tid] , rvy[tid] , rvz[tid]}; //This line creates a 3-element array RV to store the current relative velocity components (rvx[tid], rvy[tid], rvz[tid]) of the current particle.
        double rv[3] = {0.};//This line creates a 3-element array rv initialized to all zeros. This array will be used to store the updated relative velocity components after applying the rotation.
        
        //The following two nested loops are used to calculate the updated relative velocity components (rv) after applying the rotation:
        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
                rv[i] += rot[idxx*9+3*j+i] * RV[j];//This line updates the i-th component of rv by adding the product of the corresponding element from the rotation matrix (rot) and the j-th component of the current relative velocity (RV[j]).
                                                   //The rotation matrix element to be used is rot[idxx*9+3*j+i]. Since rot is stored as a 1D array representing a 3x3 matrix for each cell,
                                                   //the index calculation idxx*9+3*j+i accesses the appropriate element of the rotation matrix for the current cell. 
                                                   //The i index iterates over rows, and the j index iterates over columns, effectively performing matrix multiplication between the rotation matrix and the relative velocity vector.
        }
    
        // This line updates the relative velocity components (rvx[tid], rvy[tid], rvz[tid]) of the current particle with the values stored in the rv array.
        // These updated relative velocity components now reflect the rotation applied to the particle's motion.
        rvx[tid] = rv[0];
        rvy[tid] = rv[1];
        rvz[tid] = rv[2];   
    }
} //Output: The rvx, rvy, and rvz arrays will be updated with the transformed velocities.

__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n,int* m,int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0;
        uy[tid] = 0;
        uz[tid] = 0;
        n[tid] = 0;
        e[tid]=0;
        m[tid] = 0;
    }

}

__global__ void UpdateVelocity(double* vx, double *vy, double *vz , double* ux, double *uy , double *uz ,double *factor,int *index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;// This index identifies which particle's velocity components (vx, vy, vz) the current thread will handle.
    if (tid<N)
    {   
        //idxx represents the unique ID of the cell associated with the current particle. It is used to access the corresponding mean velocities of the cell (ux[idxx], uy[idxx], uz[idxx]) 
        //in order to update the particle velocities (vx[tid], vy[tid], vz[tid]) based on the mean velocities and the factor array.
        const unsigned int idxx = index[tid];
        vx[tid] = ux[idxx] + vx[tid]*factor[idxx]; 
        vy[tid] = uy[idxx] + vy[tid]*factor[idxx];
        vz[tid] = uz[idxx] + vz[tid]*factor[idxx];
    }

}

__global__ void relativeVelocity(double* ux , double* uy , double* uz, int* n, double* vx, double* vy, double* vz, int* index,int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        vx[tid] = vx[tid] - ux[idxx] ;
        vy[tid] = vy[tid] - uy[idxx] ;
        vz[tid] = vz[tid] - uz[idxx] ;
    }

}

__host__ void MPCD_MD_collision(double* d_vx ,double*  d_vy ,double*  d_vz , int* d_index,
double* d_mdVx ,double*  d_mdVy,double*  d_mdVz , int *d_mdIndex,
double* d_ux ,double*  d_uy ,double*  d_uz ,
double *d_e ,double *d_scalefactor, int *d_n , int* d_m,
double *d_rot, double *d_theta, double *d_phi ,
int N , int Nmd, int Nc,
curandState *devStates, int grid_size)
{
            //This launches the MakeCellReady kernel with the specified grid size and block size.
            //The kernel resets cell properties such as mean velocity (d_ux, d_uy, d_uz), energy (d_e), and count (d_n, d_m) to zero for all cells (Nc).
            MakeCellReady<<<grid_size,blockSize>>>(d_ux , d_uy, d_uz ,d_e, d_n,d_m,Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );            

            //This launches the MeanVelCell kernel with the specified grid size and block size.
            //The kernel calculates the mean velocities (d_ux, d_uy, d_uz) of particles within each cell based on their individual velocities (d_vx, d_vy, d_vz). 
            //The d_index array maps each particle to its corresponding cell. 
            //The result is updated in the d_ux, d_uy, and d_uz arrays, and the particle count (d_n) and mass (d_m) arrays are updated for each cell (N is the total number of particles).
            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_vx , d_uy, d_vy, d_uz, d_vz, d_index, d_n , d_m, 1 ,N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //This launches the MeanVelCell kernel again, but this time it calculates the mean velocities of MD particles within each cell.
            // The MD particle velocities are provided in the d_mdVx, d_mdVy, and d_mdVz arrays, and the d_mdIndex array maps each MD particle to its corresponding cell.
            // The result is updated in the d_ux, d_uy, and d_uz arrays, and the particle count (d_n) and mass (d_m) arrays are updated for each MD cell (Nmd is the total number of MD particles).
            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_mdVx , d_uy, d_mdVy, d_uz, d_mdVz, d_mdIndex, d_n ,d_m , density ,Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() ); 

            //This launches the RotationStep1 kernel with the specified grid size and block size.
            // The kernel calculates the rotation matrices (d_rot) for each cell based on the angle values (d_phi, d_theta) and the mass (d_m) of particles in each cell.
            // The number of cells is given by Nc.
            RotationStep1<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_rot, d_m, d_phi, d_theta, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // The kernel calculates the relative velocities between particles and their corresponding cell mean velocities.
            // It uses the previously computed mean velocities (d_ux, d_uy, d_uz) and particle velocities (d_vx, d_vy, d_vz). The d_index array maps each particle to its corresponding cell. 
            //The total number of particles is given by N.
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_vx, d_vy, d_vz, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //This launches the relativeVelocity kernel again, but this time it calculates the relative velocities between MPCD particles and their corresponding cell mean velocities
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //The kernel is responsible for updating the velocities of regular particles (d_vx, d_vy, d_vz) based on the calculated rotation matrices (d_rot).
            //The d_index array maps each particle to its corresponding cell, and the total number of particles is given by N.
            RotationStep2<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_rot, d_index, N);
            //This line checks for any errors that might have occurred during the kernel launch using the cudaPeekAtLastError() function. If there are any errors, they will be recorded, and the error status will be reset for the next kernel launch.
            gpuErrchk( cudaPeekAtLastError() );
            //This line synchronizes the device and the host. It ensures that all previously issued CUDA calls are completed before continuing with the program execution. This synchronization is needed because the subsequent operations may depend on the results of the previous kernel execution.
            gpuErrchk( cudaDeviceSynchronize() );


            //Similar to the previous line, this one launches the RotationStep2 kernel again. 
            //However, this time it updates the velocities of MD particles (d_mdVx, d_mdVy, d_mdVz) based on the calculated rotation matrices (d_rot). The d_mdIndex array maps each MD particle to its corresponding cell, and the total number of MD particles is given by Nmd.
            RotationStep2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_rot, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //The kernel is responsible for updating the cell energy (d_e) due to the velocity changes of regular particles. It uses the updated particle velocities (d_vx, d_vy, d_vz) and the d_index array that maps each particle to its corresponding cell. The total number of particles is given by N, 
            //and the last argument 1 is likely a parameter specifying the mass of particles.
            E_cell<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_e, d_index, N, 1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            // Similar to the previous line, this one launches the E_cell kernel again. However, this time it updates the cell energy (d_e) due to the velocity changes of MD particles
            //The total number of MD particles is given by Nmd
            E_cell<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_e, d_mdIndex, Nmd , density);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            MBS<<<grid_size,blockSize>>>(d_scalefactor,d_n,d_e,devStates, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_ux, d_uy, d_uz, d_scalefactor, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_ux, d_uy, d_uz, d_scalefactor, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
}
