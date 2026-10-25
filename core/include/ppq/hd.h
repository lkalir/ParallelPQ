#ifndef PPQ_HD_H_
#define PPQ_HD_H_

#if defined(__HIPCC__) || defined(__CUDACC__)
#define PPQ_HD __host__ __device__
#else
#define PPQ_HD
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PPQ_UNROLL _Pragma("unroll")
#else
#define PPQ_UNROLL
#endif

#endif
