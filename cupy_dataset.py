import cupy
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
cupy.cuda.set_allocator(None)

cupy_batched_barrier_option = cupy.RawKernel(r'''
extern "C" __global__ void batched_barrier_option(
    float *d_s,
    const float T,
    const float * K,
    const float * B,
    const float * S0,
    const float * sigma,
    const float * mu,
    const float * r,
    const float * d_normals,
    const long N_STEPS,
    const long N_PATHS,
    const long N_BATCH)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;
  const float tmp3 = sqrt(T/N_STEPS);


  for (unsigned i = idx; i<N_PATHS * N_BATCH; i+=stride)
  {
    int batch_id = i / N_PATHS;
    int path_id = i % N_PATHS;
    float s_curr = S0[batch_id];
    float tmp1 = mu[batch_id]*T/N_STEPS;
    float tmp2 = exp(-r[batch_id]*T);
    unsigned n=0;
    double running_average = 0.0;
    for(unsigned n = 0; n < N_STEPS; n++){
       s_curr += tmp1 * s_curr + sigma[batch_id]*s_curr*tmp3*d_normals[path_id + batch_id * N_PATHS + n * N_PATHS * N_BATCH];
       running_average += (s_curr - running_average) / (n + 1.0);
       if (running_average <= B[batch_id]){
           break;
       }
    }

    float payoff = (running_average>K[batch_id] ? running_average-K[batch_id] : 0.f); 
    d_s[i] = tmp2 * payoff;
  }
}

''', 'batched_barrier_option')

class OptionDataSet(torch.utils.data.IterableDataset):
    
    def __init__(self, max_len=10, number_path = 1000, batch=2, threads=256,seed=15):
        self.num = 0
        self.max_length = max_len
        self.N_PATHS = number_path
        self.N_STEPS = 365
        self.N_BATCH = batch
        self.T = np.float32(1.0)
        self.output = cupy.zeros(self.N_BATCH*self.N_PATHS, dtype=cupy.float32) 
        self.number_of_blocks = (self.N_PATHS * self.N_BATCH - 1) // threads + 1
        self.number_of_threads = threads
        cupy.random.seed(seed)
        
    def __len__(self):
        return self.max_length
        
    def __iter__(self):
        self.num = 0
        return self
    
    def __next__(self):
        if self.num > self.max_length:
            raise StopIteration
        X = cupy.random.rand(self.N_BATCH, 6, dtype=cupy.float32)
        # scale the [0, 1) random numbers to the correct range for each of the option parameters
        X = X * cupy.array([200.0, 0.99, 200.0, 0.4, 0.2, 0.2], dtype=cupy.float32)
        # make sure the Barrier is smaller than the Strike price
        X[:, 1] = X[:, 0] * X[:, 1]
        randoms = cupy.random.normal(0, 1, self.N_BATCH * self.N_PATHS * self.N_STEPS, dtype=cupy.float32)
        cupy_batched_barrier_option((self.number_of_blocks,), (self.number_of_threads,), (self.output, self.T, cupy.ascontiguousarray(X[:, 0]), 
                              cupy.ascontiguousarray(X[:, 1]), cupy.ascontiguousarray(X[:, 2]), cupy.ascontiguousarray(X[:, 3]), cupy.ascontiguousarray(X[:, 4]), cupy.ascontiguousarray(X[:, 5]), randoms, self.N_STEPS, self.N_PATHS, self.N_BATCH))
        Y = self.output.reshape(self.N_BATCH, self.N_PATHS).mean(axis=1)
        self.num += 1
        return (from_dlpack(X.toDlpack()), from_dlpack(Y.toDlpack()))