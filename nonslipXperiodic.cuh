//no slip boundary condition 

__device__ double heaviside_left(double x, double L){


    if (x < L)
        return 0.0;
    else if (x > L)
        return 1.0;
    else
        return 0.0;
}

__device__ double heaviside_right(double x, double L){


    if (x < L)
        return 0.0;
    else if (x > L)
        return 1.0;
    else
        return 1.0;
}

__device__ double symmetric_heaviside_left(double x, double L){


    if (x < L)
        return -1.0;
    else if (x > L)
        return 1.0;
    else
        return -1.0;
}

__device__ double symmetric_heaviside_right(double x, double L){


    if (x < L)
        return -1.0;
    else if (x > L)
        return 1.0;
    else
        return 1.0;

}

__global__ void nonslipXperiodicBC1(double *x1 ,double *x2 , double *x3, double *v1 ,double *v2, double *v3, double ux,double *L, double t, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N)
    {

        //check to see if the particle is in y=-L[1]/2 or y=L[1]/2 or z=-L[2]/2 or z=L[2]/2 planes (cube sides)
       

            //use the heaviside_right and heaviside_left functions in nonslipXperiodicBC kernel.
           
            v2[tid] *= (heaviside_left(x2[tid],L[1]/2)-heaviside_right(x2[tid],-L[1]/2));// vy in y plane
            v3[tid] *= (heaviside_left(x3[tid],L[2]/2)-heaviside_right(x3[tid],-L[2]/2));// vz in z plane
           
            x1[tid] -= ux * t * round(x3[tid] / L[2]);
            x1[tid] -= L[0] * (round(x1[tid] / L[0]));
            v1[tid] -= ux * round(x3[tid] / L[2]);
            x2[tid] -= L[1] * (round(x2[tid] / L[1]));
            x3[tid] -= L[2] * (round(x3[tid] / L[2]));
       
      
    }
}

