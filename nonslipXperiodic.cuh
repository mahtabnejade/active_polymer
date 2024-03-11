//no slip boundary condition 


__global__ void nonslipXperiodicBC(double *x1 ,double *x2 , double *x3, double *v1 ,double *v2, double *v3, double ux,double *L, double t, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N)
    {

        //check to see if the particle is in y=L[1] or y=0 or z=L[2] or z=0 planes (cube sides)
        if (x2[tid] == 0 || x2[tid] == L[1] || x3[tid] == 0 || x3[tid] == L[2]){

            
            v2[tid] = 0.0;
            v3[tid] = 0.0;
            x1[tid] -= ux * t * round(x3[tid] / L[2]);
            x1[tid] -= L[0] * (round(x1[tid] / L[0]));
            v1[tid] -= ux * round(x3[tid] / L[2]);
        }
        else{
        x1[tid] -= ux * t * round(x3[tid] / L[2]);
        x1[tid] -= L[0] * (round(x1[tid] / L[0]));
        v1[tid] -= ux * round(x3[tid] / L[2]);
        }
      
    }
}
