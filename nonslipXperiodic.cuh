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

__device__ double XL(double x, double L, double e){

    if (x > (L - e))
        return L;
    else  if (x == (L - e))
        return L;
    else if ((-L + e) < x < (L - e))
        return x;
    else 
        return -L;
    

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

//this function inverses the direction of particles when they reach the boundaries while keeping the perpendicular velocities on walls equal to zero. 
//in this function we multipy the tangential components of velocity on walls by -1. 
__global__ void nonslipXperiodicBC2(double *x1 ,double *x2 , double *x3, double *v1 ,double *v2, double *v3, double ux,double *L, double t, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N)
    {

        //check to see if the particle is in y=-L[1]/2 or y=L[1]/2 or z=-L[2]/2 or z=L[2]/2 planes (cube sides)
       

            //use the heaviside_right and heaviside_left functions in nonslipXperiodicBC kernel.

            v2[tid] *= ((symmetric_heaviside_left(x2[tid],-L[1]/2)-symmetric_heaviside_right(x2[tid],L[1]/2)) - 1);// vy in y plane (in cube sides must be multipied by -1 but elsewhere must be multipied by 1 )
            v3[tid] *= ((symmetric_heaviside_left(x3[tid],-L[2]/2)-symmetric_heaviside_right(x3[tid],L[2]/2)) - 1);// vz in z plane (in cube sides must be multipied by -1 but elsewhere must be multipied by 1 )
           //we could also go like this:
            //v2[tid] *= ((2*(heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2)))-1);
            //v3[tid] *= ((2*(heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2)))-1);

           
            v1[tid] *= (heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2));// vx in y plane (in cube sides should be zero but elsewhere must be multipied by 1)
            

            v3[tid] *= (heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2));// vz in y plane  (in cube sides should be zero but elsewhere must be multipied by 1 )
            
            v1[tid] *= (heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2));// vx in z plane (in cube sides should be zero but elsewhere must be multipied by 1)
            v2[tid] *= (heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2));// vy in z plane (in cube sides should be zero but elsewhere must be multipied by 1)
           //we keep the x plane periodic still. 
            x1[tid] -= ux * t * round(x3[tid] / L[2]);
            x1[tid] -= L[0] * (round(x1[tid] / L[0]));
            v1[tid] -= ux * round(x3[tid] / L[2]);
            
    }
}

//a function to account for a situation where all three components of velocity become zero on walls.
__global__ void nonslipXperiodicBC3(double *x1 ,double *x2 , double *x3, double *v1 ,double *v2, double *v3, double ux,double *L, double t, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N)
    {

        //check to see if the particle is in y=-L[1]/2 or y=L[1]/2 or z=-L[2]/2 or z=L[2]/2 planes (cube sides)
       

            //use the heaviside_right and heaviside_left functions in nonslipXperiodicBC kernel.

            v2[tid] *= (heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2));// vy in y plane (in cube sides must be zero but elsewhere must be multipied by 1 )
            v1[tid] *= ((heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2)) );// vx in y plane (in cube sides must be zero but elsewhere must be multipied by 1)
            v3[tid] *= ((heaviside_left(x2[tid],-L[1]/2)-heaviside_right(x2[tid],L[1]/2)) );// vz in y plane  (in cube sides must be zero but elsewhere must be multipied by 1 )
            
            v1[tid] *= ((heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2)) );// vx in z plane (in cube sides must be zero but elsewhere must be multipied by 1)
            v2[tid] *= ((heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2)) );// vy in z plane (in cube sides must be zero but elsewhere must be multipied by 1)
            v3[tid] *= (heaviside_left(x3[tid],-L[2]/2)-heaviside_right(x3[tid],L[2]/2));// vz in z plane (in cube sides must be zero but elsewhere must be multipied by 1 )


           //we keep the x plane periodic still. 
            x1[tid] -= ux * t * round(x3[tid] / L[2]);
            x1[tid] -= L[0] * (round(x1[tid] / L[0]));
            v1[tid] -= ux * round(x3[tid] / L[2]);

        //we should make sure at the same time the particles are exactly on the walls when they velocities become zero
            double epsilon = 0.3;
            x2[tid] = XL(x2[tid] , L[1]/2, epsilon);
            x3[tid] = XL(x3[tid] , L[2]/2, epsilon);


            
    }
}
