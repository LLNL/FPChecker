# 1 "comp.cu"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 1
# 56 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push


#pragma GCC diagnostic ignored "-Wunused-function"
# 78 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_config.h" 1
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_config.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/host_config.h" 1
# 179 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/host_config.h"
# 1 "/usr/include/features.h" 1 3 4
# 375 "/usr/include/features.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 392 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 393 "/usr/include/sys/cdefs.h" 2 3 4
# 376 "/usr/include/features.h" 2 3 4
# 399 "/usr/include/features.h" 3 4
# 1 "/usr/include/gnu/stubs.h" 1 3 4




# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 6 "/usr/include/gnu/stubs.h" 2 3 4
# 14 "/usr/include/gnu/stubs.h" 3 4
# 1 "/usr/include/gnu/stubs-64-v2.h" 1 3 4
# 15 "/usr/include/gnu/stubs.h" 2 3 4
# 400 "/usr/include/features.h" 2 3 4
# 180 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/host_config.h" 2
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_config.h" 2
# 79 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2







# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 56 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_types.h" 1
# 53 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/host_defines.h" 1
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 2
# 54 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_types.h" 2







enum __attribute__((device_builtin)) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};
# 57 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 2


# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h" 1
# 53 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 54 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h" 1
# 59 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 60 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h" 2
# 93 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h"
struct __attribute__((device_builtin)) char1
{
    signed char x;
};

struct __attribute__((device_builtin)) uchar1
{
    unsigned char x;
};


struct __attribute__((device_builtin)) __attribute__((aligned(2))) char2
{
    signed char x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2))) uchar2
{
    unsigned char x, y;
};

struct __attribute__((device_builtin)) char3
{
    signed char x, y, z;
};

struct __attribute__((device_builtin)) uchar3
{
    unsigned char x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) char4
{
    signed char x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) uchar4
{
    unsigned char x, y, z, w;
};

struct __attribute__((device_builtin)) short1
{
    short x;
};

struct __attribute__((device_builtin)) ushort1
{
    unsigned short x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) short2
{
    short x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) ushort2
{
    unsigned short x, y;
};

struct __attribute__((device_builtin)) short3
{
    short x, y, z;
};

struct __attribute__((device_builtin)) ushort3
{
    unsigned short x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __attribute__((device_builtin)) int1
{
    int x;
};

struct __attribute__((device_builtin)) uint1
{
    unsigned int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) int2 { int x; int y; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };

struct __attribute__((device_builtin)) int3
{
    int x, y, z;
};

struct __attribute__((device_builtin)) uint3
{
    unsigned int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) int4
{
    int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) uint4
{
    unsigned int x, y, z, w;
};

struct __attribute__((device_builtin)) long1
{
    long int x;
};

struct __attribute__((device_builtin)) ulong1
{
    unsigned long x;
};






struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(long int)))) long2
{
    long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
    unsigned long int x, y;
};



struct __attribute__((device_builtin)) long3
{
    long int x, y, z;
};

struct __attribute__((device_builtin)) ulong3
{
    unsigned long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) long4
{
    long int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulong4
{
    unsigned long int x, y, z, w;
};

struct __attribute__((device_builtin)) float1
{
    float x;
};
# 269 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h"
struct __attribute__((device_builtin)) __attribute__((aligned(8))) float2 { float x; float y; };




struct __attribute__((device_builtin)) float3
{
    float x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) float4
{
    float x, y, z, w;
};

struct __attribute__((device_builtin)) longlong1
{
    long long int x;
};

struct __attribute__((device_builtin)) ulonglong1
{
    unsigned long long int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong2
{
    long long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong2
{
    unsigned long long int x, y;
};

struct __attribute__((device_builtin)) longlong3
{
    long long int x, y, z;
};

struct __attribute__((device_builtin)) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong4
{
    long long int x, y, z ,w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __attribute__((device_builtin)) double1
{
    double x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double2
{
    double x, y;
};

struct __attribute__((device_builtin)) double3
{
    double x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double4
{
    double x, y, z, w;
};
# 356 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_types.h"
typedef __attribute__((device_builtin)) struct char1 char1;
typedef __attribute__((device_builtin)) struct uchar1 uchar1;
typedef __attribute__((device_builtin)) struct char2 char2;
typedef __attribute__((device_builtin)) struct uchar2 uchar2;
typedef __attribute__((device_builtin)) struct char3 char3;
typedef __attribute__((device_builtin)) struct uchar3 uchar3;
typedef __attribute__((device_builtin)) struct char4 char4;
typedef __attribute__((device_builtin)) struct uchar4 uchar4;
typedef __attribute__((device_builtin)) struct short1 short1;
typedef __attribute__((device_builtin)) struct ushort1 ushort1;
typedef __attribute__((device_builtin)) struct short2 short2;
typedef __attribute__((device_builtin)) struct ushort2 ushort2;
typedef __attribute__((device_builtin)) struct short3 short3;
typedef __attribute__((device_builtin)) struct ushort3 ushort3;
typedef __attribute__((device_builtin)) struct short4 short4;
typedef __attribute__((device_builtin)) struct ushort4 ushort4;
typedef __attribute__((device_builtin)) struct int1 int1;
typedef __attribute__((device_builtin)) struct uint1 uint1;
typedef __attribute__((device_builtin)) struct int2 int2;
typedef __attribute__((device_builtin)) struct uint2 uint2;
typedef __attribute__((device_builtin)) struct int3 int3;
typedef __attribute__((device_builtin)) struct uint3 uint3;
typedef __attribute__((device_builtin)) struct int4 int4;
typedef __attribute__((device_builtin)) struct uint4 uint4;
typedef __attribute__((device_builtin)) struct long1 long1;
typedef __attribute__((device_builtin)) struct ulong1 ulong1;
typedef __attribute__((device_builtin)) struct long2 long2;
typedef __attribute__((device_builtin)) struct ulong2 ulong2;
typedef __attribute__((device_builtin)) struct long3 long3;
typedef __attribute__((device_builtin)) struct ulong3 ulong3;
typedef __attribute__((device_builtin)) struct long4 long4;
typedef __attribute__((device_builtin)) struct ulong4 ulong4;
typedef __attribute__((device_builtin)) struct float1 float1;
typedef __attribute__((device_builtin)) struct float2 float2;
typedef __attribute__((device_builtin)) struct float3 float3;
typedef __attribute__((device_builtin)) struct float4 float4;
typedef __attribute__((device_builtin)) struct longlong1 longlong1;
typedef __attribute__((device_builtin)) struct ulonglong1 ulonglong1;
typedef __attribute__((device_builtin)) struct longlong2 longlong2;
typedef __attribute__((device_builtin)) struct ulonglong2 ulonglong2;
typedef __attribute__((device_builtin)) struct longlong3 longlong3;
typedef __attribute__((device_builtin)) struct ulonglong3 ulonglong3;
typedef __attribute__((device_builtin)) struct longlong4 longlong4;
typedef __attribute__((device_builtin)) struct ulonglong4 ulonglong4;
typedef __attribute__((device_builtin)) struct double1 double1;
typedef __attribute__((device_builtin)) struct double2 double2;
typedef __attribute__((device_builtin)) struct double3 double3;
typedef __attribute__((device_builtin)) struct double4 double4;







struct __attribute__((device_builtin)) dim3
{
    unsigned int x, y, z;

    __attribute__((host)) __attribute__((device)) dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __attribute__((host)) __attribute__((device)) dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __attribute__((host)) __attribute__((device)) operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

};

typedef __attribute__((device_builtin)) struct dim3 dim3;
# 55 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h" 2
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 1 3 4
# 34 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 3 4
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/syslimits.h" 1 3 4






# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 1 3 4
# 168 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 3 4
# 1 "/usr/include/limits.h" 1 3 4
# 144 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/posix1_lim.h" 1 3 4
# 160 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/local_lim.h" 1 3 4
# 39 "/usr/include/bits/local_lim.h" 3 4
# 1 "/usr/include/linux/limits.h" 1 3 4
# 40 "/usr/include/bits/local_lim.h" 2 3 4
# 161 "/usr/include/bits/posix1_lim.h" 2 3 4
# 145 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/posix2_lim.h" 1 3 4
# 149 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/xopen_lim.h" 1 3 4
# 33 "/usr/include/bits/xopen_lim.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 34 "/usr/include/bits/xopen_lim.h" 2 3 4
# 153 "/usr/include/limits.h" 2 3 4
# 169 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 2 3 4
# 8 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/syslimits.h" 2 3 4
# 35 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include-fixed/limits.h" 2 3 4
# 73 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h" 2
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 147 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 3 4
typedef long int ptrdiff_t;
# 212 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 74 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h" 2
# 171 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
enum __attribute__((device_builtin)) cudaError
{





    cudaSuccess = 0,





    cudaErrorMissingConfiguration = 1,





    cudaErrorMemoryAllocation = 2,





    cudaErrorInitializationError = 3,
# 206 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorLaunchFailure = 4,
# 215 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorPriorLaunchFailure = 5,
# 226 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorLaunchTimeout = 6,
# 235 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorLaunchOutOfResources = 7,





    cudaErrorInvalidDeviceFunction = 8,
# 250 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorInvalidConfiguration = 9,





    cudaErrorInvalidDevice = 10,





    cudaErrorInvalidValue = 11,





    cudaErrorInvalidPitchValue = 12,





    cudaErrorInvalidSymbol = 13,




    cudaErrorMapBufferObjectFailed = 14,




    cudaErrorUnmapBufferObjectFailed = 15,





    cudaErrorInvalidHostPointer = 16,





    cudaErrorInvalidDevicePointer = 17,





    cudaErrorInvalidTexture = 18,





    cudaErrorInvalidTextureBinding = 19,






    cudaErrorInvalidChannelDescriptor = 20,





    cudaErrorInvalidMemcpyDirection = 21,
# 331 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorAddressOfConstant = 22,
# 340 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorTextureFetchFailed = 23,
# 349 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorTextureNotBound = 24,
# 358 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorSynchronizationError = 25,





    cudaErrorInvalidFilterSetting = 26,





    cudaErrorInvalidNormSetting = 27,







    cudaErrorMixedDeviceExecution = 28,






    cudaErrorCudartUnloading = 29,




    cudaErrorUnknown = 30,







    cudaErrorNotYetImplemented = 31,
# 407 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorMemoryValueTooLarge = 32,






    cudaErrorInvalidResourceHandle = 33,







    cudaErrorNotReady = 34,






    cudaErrorInsufficientDriver = 35,
# 442 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorSetOnActiveProcess = 36,





    cudaErrorInvalidSurface = 37,





    cudaErrorNoDevice = 38,





    cudaErrorECCUncorrectable = 39,




    cudaErrorSharedObjectSymbolNotFound = 40,




    cudaErrorSharedObjectInitFailed = 41,





    cudaErrorUnsupportedLimit = 42,





    cudaErrorDuplicateVariableName = 43,





    cudaErrorDuplicateTextureName = 44,





    cudaErrorDuplicateSurfaceName = 45,
# 504 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorDevicesUnavailable = 46,




    cudaErrorInvalidKernelImage = 47,







    cudaErrorNoKernelImageForDevice = 48,
# 530 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorIncompatibleDriverContext = 49,






    cudaErrorPeerAccessAlreadyEnabled = 50,






    cudaErrorPeerAccessNotEnabled = 51,





    cudaErrorDeviceAlreadyInUse = 54,






    cudaErrorProfilerDisabled = 55,







    cudaErrorProfilerNotInitialized = 56,






    cudaErrorProfilerAlreadyStarted = 57,






     cudaErrorProfilerAlreadyStopped = 58,







    cudaErrorAssert = 59,






    cudaErrorTooManyPeers = 60,





    cudaErrorHostMemoryAlreadyRegistered = 61,





    cudaErrorHostMemoryNotRegistered = 62,




    cudaErrorOperatingSystem = 63,





    cudaErrorPeerAccessUnsupported = 64,






    cudaErrorLaunchMaxDepthExceeded = 65,







    cudaErrorLaunchFileScopedTex = 66,







    cudaErrorLaunchFileScopedSurf = 67,
# 655 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorSyncDepthExceeded = 68,
# 667 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorLaunchPendingCountExceeded = 69,




    cudaErrorNotPermitted = 70,





    cudaErrorNotSupported = 71,
# 687 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorHardwareStackError = 72,







    cudaErrorIllegalInstruction = 73,
# 704 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorMisalignedAddress = 74,
# 715 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorInvalidAddressSpace = 75,







    cudaErrorInvalidPc = 76,







    cudaErrorIllegalAddress = 77,





    cudaErrorInvalidPtx = 78,




    cudaErrorInvalidGraphicsContext = 79,





    cudaErrorNvlinkUncorrectable = 80,






    cudaErrorJitCompilerNotFound = 81,
# 764 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    cudaErrorCooperativeLaunchTooLarge = 82,




    cudaErrorStartupFailure = 0x7f,







    cudaErrorApiFailureBase = 10000
};




enum __attribute__((device_builtin)) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3
};




struct __attribute__((device_builtin)) cudaChannelFormatDesc
{
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;




enum __attribute__((device_builtin)) cudaMemoryType
{
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2
};




enum __attribute__((device_builtin)) cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};






struct __attribute__((device_builtin)) cudaPitchedPtr
{
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};






struct __attribute__((device_builtin)) cudaExtent
{
    size_t width;
    size_t height;
    size_t depth;
};






struct __attribute__((device_builtin)) cudaPos
{
    size_t x;
    size_t y;
    size_t z;
};




struct __attribute__((device_builtin)) cudaMemcpy3DParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;

    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};




struct __attribute__((device_builtin)) cudaMemcpy3DPeerParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;

    struct cudaExtent extent;
};




struct cudaGraphicsResource;




enum __attribute__((device_builtin)) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8
};




enum __attribute__((device_builtin)) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2
};




enum __attribute__((device_builtin)) cudaGraphicsCubeFace
{
    cudaGraphicsCubeFacePositiveX = 0x00,
    cudaGraphicsCubeFaceNegativeX = 0x01,
    cudaGraphicsCubeFacePositiveY = 0x02,
    cudaGraphicsCubeFaceNegativeY = 0x03,
    cudaGraphicsCubeFacePositiveZ = 0x04,
    cudaGraphicsCubeFaceNegativeZ = 0x05
};




enum __attribute__((device_builtin)) cudaResourceType
{
    cudaResourceTypeArray = 0x00,
    cudaResourceTypeMipmappedArray = 0x01,
    cudaResourceTypeLinear = 0x02,
    cudaResourceTypePitch2D = 0x03
};




enum __attribute__((device_builtin)) cudaResourceViewFormat
{
    cudaResViewFormatNone = 0x00,
    cudaResViewFormatUnsignedChar1 = 0x01,
    cudaResViewFormatUnsignedChar2 = 0x02,
    cudaResViewFormatUnsignedChar4 = 0x03,
    cudaResViewFormatSignedChar1 = 0x04,
    cudaResViewFormatSignedChar2 = 0x05,
    cudaResViewFormatSignedChar4 = 0x06,
    cudaResViewFormatUnsignedShort1 = 0x07,
    cudaResViewFormatUnsignedShort2 = 0x08,
    cudaResViewFormatUnsignedShort4 = 0x09,
    cudaResViewFormatSignedShort1 = 0x0a,
    cudaResViewFormatSignedShort2 = 0x0b,
    cudaResViewFormatSignedShort4 = 0x0c,
    cudaResViewFormatUnsignedInt1 = 0x0d,
    cudaResViewFormatUnsignedInt2 = 0x0e,
    cudaResViewFormatUnsignedInt4 = 0x0f,
    cudaResViewFormatSignedInt1 = 0x10,
    cudaResViewFormatSignedInt2 = 0x11,
    cudaResViewFormatSignedInt4 = 0x12,
    cudaResViewFormatHalf1 = 0x13,
    cudaResViewFormatHalf2 = 0x14,
    cudaResViewFormatHalf4 = 0x15,
    cudaResViewFormatFloat1 = 0x16,
    cudaResViewFormatFloat2 = 0x17,
    cudaResViewFormatFloat4 = 0x18,
    cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
    cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
    cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
    cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
    cudaResViewFormatSignedBlockCompressed4 = 0x1d,
    cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
    cudaResViewFormatSignedBlockCompressed5 = 0x1f,
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
    cudaResViewFormatSignedBlockCompressed6H = 0x21,
    cudaResViewFormatUnsignedBlockCompressed7 = 0x22
};




struct __attribute__((device_builtin)) cudaResourceDesc {
 enum cudaResourceType resType;

 union {
  struct {
   cudaArray_t array;
  } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
  struct {
   void *devPtr;
   struct cudaChannelFormatDesc desc;
   size_t sizeInBytes;
  } linear;
  struct {
   void *devPtr;
   struct cudaChannelFormatDesc desc;
   size_t width;
   size_t height;
   size_t pitchInBytes;
  } pitch2D;
 } res;
};




struct __attribute__((device_builtin)) cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};




struct __attribute__((device_builtin)) cudaPointerAttributes
{




    enum cudaMemoryType memoryType;
# 1076 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
    int device;





    void *devicePointer;





    void *hostPointer;




    int isManaged;
};




struct __attribute__((device_builtin)) cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;





   int cacheModeCA;






   int maxDynamicSharedSizeBytes;
# 1165 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
   int preferredShmemCarveout;
};




enum __attribute__((device_builtin)) cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeMax
};




enum __attribute__((device_builtin)) cudaFuncCache
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};





enum __attribute__((device_builtin)) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __attribute__((device_builtin)) cudaSharedCarveout {
    cudaSharedmemCarveoutDefault = -1,
    cudaSharedmemCarveoutMaxShared = 100,
    cudaSharedmemCarveoutMaxL1 = 0
};




enum __attribute__((device_builtin)) cudaComputeMode
{
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};




enum __attribute__((device_builtin)) cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04
};




enum __attribute__((device_builtin)) cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6
};




enum __attribute__((device_builtin)) cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4
};




enum __attribute__((device_builtin)) cudaOutputMode
{
    cudaKeyValuePair = 0x00,
    cudaCSV = 0x01
};




enum __attribute__((device_builtin)) cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101
};





enum __attribute__((device_builtin)) cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4
};



struct __attribute__((device_builtin)) cudaDeviceProp
{
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
};
# 1547 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcMemHandle_st
{
    char reserved[64];
}cudaIpcMemHandle_t;
# 1569 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_types.h"
typedef __attribute__((device_builtin)) enum cudaError cudaError_t;




typedef __attribute__((device_builtin)) struct CUstream_st *cudaStream_t;




typedef __attribute__((device_builtin)) struct CUevent_st *cudaEvent_t;




typedef __attribute__((device_builtin)) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __attribute__((device_builtin)) struct CUuuid_st cudaUUID_t;




typedef __attribute__((device_builtin)) enum cudaOutputMode cudaOutputMode_t;




enum __attribute__((device_builtin)) cudaCGScope {
    cudaCGScopeInvalid = 0,
    cudaCGScopeGrid = 1,
    cudaCGScopeMultiGrid = 2
};




struct __attribute__((device_builtin)) cudaLaunchParams
{
    void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
};
# 60 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 2


# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_types.h" 1
# 84 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_types.h"
enum __attribute__((device_builtin)) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2
};




enum __attribute__((device_builtin)) cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1
};




struct __attribute__((device_builtin)) surfaceReference
{



    struct cudaChannelFormatDesc channelDesc;
};




typedef __attribute__((device_builtin)) unsigned long long cudaSurfaceObject_t;
# 63 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_types.h" 1
# 84 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_types.h"
enum __attribute__((device_builtin)) cudaTextureAddressMode
{
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
};




enum __attribute__((device_builtin)) cudaTextureFilterMode
{
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
};




enum __attribute__((device_builtin)) cudaTextureReadMode
{
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
};




struct __attribute__((device_builtin)) textureReference
{



    int normalized;



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureAddressMode addressMode[3];



    struct cudaChannelFormatDesc channelDesc;



    int sRGB;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;
    int __cudaReserved[15];
};




struct __attribute__((device_builtin)) cudaTextureDesc
{



    enum cudaTextureAddressMode addressMode[3];



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureReadMode readMode;



    int sRGB;



    float borderColor[4];



    int normalizedCoords;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;
};




typedef __attribute__((device_builtin)) unsigned long long cudaTextureObject_t;
# 64 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 2
# 87 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/library_types.h" 1
# 54 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/library_types.h"
typedef enum cudaDataType_t
{
 CUDA_R_16F= 2,
 CUDA_C_16F= 6,
 CUDA_R_32F= 0,
 CUDA_C_32F= 4,
 CUDA_R_64F= 1,
 CUDA_C_64F= 5,
 CUDA_R_8I = 3,
 CUDA_C_8I = 7,
 CUDA_R_8U = 8,
 CUDA_C_8U = 9,
 CUDA_R_32I= 10,
 CUDA_C_32I= 11,
 CUDA_R_32U= 12,
 CUDA_C_32U= 13
} cudaDataType;


typedef enum libraryPropertyType_t
{
 MAJOR_VERSION,
 MINOR_VERSION,
 PATCH_LEVEL
} libraryPropertyType;
# 88 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2


# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h" 1
# 133 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 134 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 135 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h" 1
# 64 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h"
extern "C" {


struct cudaFuncAttributes;







__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaMalloc(void **p, size_t s)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}




}
# 119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 120 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h" 2

extern "C"
{
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceSynchronize(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetLastError(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaPeekAtLastError(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) const char* cudaGetErrorString(cudaError_t error);
extern __attribute__((device)) __attribute__((cudart_builtin)) const char* cudaGetErrorName(cudaError_t error);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceCount(int *count);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetDevice(int *device);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamDestroy(cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventDestroy(cudaEvent_t event);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFree(void *devPtr);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(void **devPtr, size_t size);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
# 178 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h"
extern __attribute__((device)) __attribute__((cudart_builtin)) void * cudaGetParameterBuffer(size_t alignment, size_t size);
# 206 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h"
extern __attribute__((device)) __attribute__((cudart_builtin)) void * cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream);
# 226 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_device_runtime_api.h"
    extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
    extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);


extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags);

extern __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long cudaCGGetIntrinsicHandle(enum cudaCGScope scope);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGGetSize(unsigned int *numThreads, unsigned int *numGrids, unsigned long long handle);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGGetRank(unsigned int *threadRank, unsigned int *gridRank, unsigned long long handle);
}

template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(T **devPtr, size_t size);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned int flags);
# 137 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h" 2
# 218 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern "C" {
# 251 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceReset(void);
# 270 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceSynchronize(void);
# 347 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
# 378 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
# 410 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
# 446 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
# 489 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
# 519 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
# 562 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
# 587 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
# 615 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
# 660 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
# 698 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
# 739 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
# 792 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
# 825 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcCloseMemHandle(void *devPtr);
# 865 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadExit(void);
# 889 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadSynchronize(void);
# 936 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
# 967 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit);
# 1002 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig);
# 1048 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
# 1104 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetLastError(void);
# 1147 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaPeekAtLastError(void);
# 1163 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) const char* cudaGetErrorName(cudaError_t error);
# 1179 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) const char* cudaGetErrorString(cudaError_t error);
# 1210 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceCount(int *count);
# 1476 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
# 1663 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
# 1701 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
# 1720 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
# 1755 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDevice(int device);
# 1774 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDevice(int *device);
# 1803 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetValidDevices(int *device_arr, int len);
# 1866 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDeviceFlags( unsigned int flags );
# 1909 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetDeviceFlags( unsigned int *flags );
# 1947 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamCreate(cudaStream_t *pStream);
# 1977 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
# 2021 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
# 2046 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
# 2069 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
# 2098 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamDestroy(cudaStream_t stream);
# 2122 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
# 2136 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
typedef void ( *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);
# 2195 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags);
# 2217 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamSynchronize(cudaStream_t stream);
# 2240 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamQuery(cudaStream_t stream);
# 2321 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length = 0, unsigned int flags = 0x04);
# 2358 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventCreate(cudaEvent_t *event);
# 2394 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
# 2432 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
# 2462 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventQuery(cudaEvent_t event);
# 2491 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventSynchronize(cudaEvent_t event);
# 2517 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventDestroy(cudaEvent_t event);
# 2559 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
# 2622 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
# 2677 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
# 2774 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags = 0);
# 2823 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
# 2878 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);
# 2913 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
# 2952 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
# 2976 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDoubleForDevice(double *d);
# 3000 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDoubleForHost(double *d);
# 3055 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
# 3099 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
# 3149 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
# 3178 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
# 3219 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunch(const void *func);
# 3338 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = 0x01);
# 3366 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(void **devPtr, size_t size);
# 3397 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocHost(void **ptr, size_t size);
# 3438 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
# 3482 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height = 0, unsigned int flags = 0);
# 3510 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFree(void *devPtr);
# 3532 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeHost(void *ptr);
# 3554 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeArray(cudaArray_t array);
# 3576 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
# 3640 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
# 3722 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
# 3743 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostUnregister(void *ptr);
# 3786 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
# 3806 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
# 3843 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
# 3980 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags = 0);
# 4117 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags = 0);
# 4144 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
# 4247 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
# 4276 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
# 4392 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream = 0);
# 4416 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream = 0);
# 4437 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
# 4461 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array);
# 4502 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
# 4535 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
# 4574 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
# 4612 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
# 4651 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
# 4697 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
# 4744 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
# 4791 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
# 4836 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
# 4877 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
# 4918 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
# 4972 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5005 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream = 0);
# 5052 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5098 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5158 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5213 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5267 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5316 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5365 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 5392 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset(void *devPtr, int value, size_t count);
# 5424 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
# 5466 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
# 5500 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream = 0);
# 5539 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0);
# 5588 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream = 0);
# 5614 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
# 5639 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
# 5707 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream = 0);
# 5821 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
# 5878 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count);
# 5915 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);
# 6069 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);
# 6108 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
# 6148 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
# 6168 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
# 6229 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
# 6262 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
# 6299 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream = 0);
# 6332 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream = 0);
# 6362 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);
# 6398 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
# 6425 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource);
# 6465 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array);
# 6501 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
# 6552 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size = (2147483647 * 2U + 1U));
# 6607 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);
# 6641 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);
# 6677 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc);
# 6699 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaUnbindTexture(const struct textureReference *texref);
# 6724 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
# 6750 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureReference(const struct textureReference **texref, const void *symbol);
# 6791 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);
# 6812 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol);
# 7040 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc);
# 7057 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
# 7075 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);
# 7093 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);
# 7112 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);
# 7153 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc);
# 7170 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
# 7187 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);
# 7216 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDriverGetVersion(int *driverVersion);
# 7235 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);




extern __attribute__((host)) cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
# 7477 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime_api.h"
}
# 63 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 64 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h" 2
# 107 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h"
template<class T> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
  return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf1(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf2(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf4(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char>(void)
{
  int e = (int)sizeof(char) * 8;


  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);



}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<signed char>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned char>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char1>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char2>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char4>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned short>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short1>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short2>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short4>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort4>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int1>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint1>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int2>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint2>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int4>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint4>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}
# 379 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/channel_descriptor.h"
template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float1>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float2>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float4>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}
# 91 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h" 1
# 53 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 54 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 55 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h" 2
# 79 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
{
  struct cudaPitchedPtr s;

  s.ptr = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;

  return s;
}
# 106 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaPos make_cudaPos(size_t x, size_t y, size_t z)
{
  struct cudaPos p;

  p.x = x;
  p.y = y;
  p.z = z;

  return p;
}
# 132 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
  struct cudaExtent e;

  e.width = w;
  e.height = h;
  e.depth = d;

  return e;
}
# 93 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2


# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 96 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h" 1
# 59 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 60 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h" 2
# 75 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h"
static __inline__ __attribute__((host)) __attribute__((device)) char1 make_char1(signed char x);

static __inline__ __attribute__((host)) __attribute__((device)) uchar1 make_uchar1(unsigned char x);

static __inline__ __attribute__((host)) __attribute__((device)) char2 make_char2(signed char x, signed char y);

static __inline__ __attribute__((host)) __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y);

static __inline__ __attribute__((host)) __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z);

static __inline__ __attribute__((host)) __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z);

static __inline__ __attribute__((host)) __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w);

static __inline__ __attribute__((host)) __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);

static __inline__ __attribute__((host)) __attribute__((device)) short1 make_short1(short x);

static __inline__ __attribute__((host)) __attribute__((device)) ushort1 make_ushort1(unsigned short x);

static __inline__ __attribute__((host)) __attribute__((device)) short2 make_short2(short x, short y);

static __inline__ __attribute__((host)) __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y);

static __inline__ __attribute__((host)) __attribute__((device)) short3 make_short3(short x,short y, short z);

static __inline__ __attribute__((host)) __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z);

static __inline__ __attribute__((host)) __attribute__((device)) short4 make_short4(short x, short y, short z, short w);

static __inline__ __attribute__((host)) __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w);

static __inline__ __attribute__((host)) __attribute__((device)) int1 make_int1(int x);

static __inline__ __attribute__((host)) __attribute__((device)) uint1 make_uint1(unsigned int x);

static __inline__ __attribute__((host)) __attribute__((device)) int2 make_int2(int x, int y);

static __inline__ __attribute__((host)) __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y);

static __inline__ __attribute__((host)) __attribute__((device)) int3 make_int3(int x, int y, int z);

static __inline__ __attribute__((host)) __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z);

static __inline__ __attribute__((host)) __attribute__((device)) int4 make_int4(int x, int y, int z, int w);

static __inline__ __attribute__((host)) __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

static __inline__ __attribute__((host)) __attribute__((device)) long1 make_long1(long int x);

static __inline__ __attribute__((host)) __attribute__((device)) ulong1 make_ulong1(unsigned long int x);

static __inline__ __attribute__((host)) __attribute__((device)) long2 make_long2(long int x, long int y);

static __inline__ __attribute__((host)) __attribute__((device)) ulong2 make_ulong2(unsigned long int x, unsigned long int y);

static __inline__ __attribute__((host)) __attribute__((device)) long3 make_long3(long int x, long int y, long int z);

static __inline__ __attribute__((host)) __attribute__((device)) ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z);

static __inline__ __attribute__((host)) __attribute__((device)) long4 make_long4(long int x, long int y, long int z, long int w);

static __inline__ __attribute__((host)) __attribute__((device)) ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w);

static __inline__ __attribute__((host)) __attribute__((device)) float1 make_float1(float x);

static __inline__ __attribute__((host)) __attribute__((device)) float2 make_float2(float x, float y);

static __inline__ __attribute__((host)) __attribute__((device)) float3 make_float3(float x, float y, float z);

static __inline__ __attribute__((host)) __attribute__((device)) float4 make_float4(float x, float y, float z, float w);

static __inline__ __attribute__((host)) __attribute__((device)) longlong1 make_longlong1(long long int x);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long int x);

static __inline__ __attribute__((host)) __attribute__((device)) longlong2 make_longlong2(long long int x, long long int y);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y);

static __inline__ __attribute__((host)) __attribute__((device)) longlong3 make_longlong3(long long int x, long long int y, long long int z);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z);

static __inline__ __attribute__((host)) __attribute__((device)) longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w);

static __inline__ __attribute__((host)) __attribute__((device)) double1 make_double1(double x);

static __inline__ __attribute__((host)) __attribute__((device)) double2 make_double2(double x, double y);

static __inline__ __attribute__((host)) __attribute__((device)) double3 make_double3(double x, double y, double z);

static __inline__ __attribute__((host)) __attribute__((device)) double4 make_double4(double x, double y, double z, double w);




# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.hpp" 1
# 59 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 60 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.hpp" 2
# 75 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.hpp"
static __inline__ __attribute__((host)) __attribute__((device)) char1 make_char1(signed char x)
{
  char1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar1 make_uchar1(unsigned char x)
{
  uchar1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char2 make_char2(signed char x, signed char y)
{
  char2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short1 make_short1(short x)
{
  short1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort1 make_ushort1(unsigned short x)
{
  ushort1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short2 make_short2(short x, short y)
{
  short2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y)
{
  ushort2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short3 make_short3(short x,short y, short z)
{
  short3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
  ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short4 make_short4(short x, short y, short z, short w)
{
  short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
  ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int1 make_int1(int x)
{
  int1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint1 make_uint1(unsigned int x)
{
  uint1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int2 make_int2(int x, int y)
{
  int2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int3 make_int3(int x, int y, int z)
{
  int3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long1 make_long1(long int x)
{
  long1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long2 make_long2(long int x, long int y)
{
  long2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long3 make_long3(long int x, long int y, long int z)
{
  long3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float1 make_float1(float x)
{
  float1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float2 make_float2(float x, float y)
{
  float2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong1 make_longlong1(long long int x)
{
  longlong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long int x)
{
  ulonglong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong2 make_longlong2(long long int x, long long int y)
{
  longlong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
  ulonglong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
  longlong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
  ulonglong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
  longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
  ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double1 make_double1(double x)
{
  double1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double4 make_double4(double x, double y, double z, double w)
{
  double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
# 175 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/vector_functions.h" 2
# 97 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 115 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/common_functions.h" 1
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/common_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/host_defines.h" 1
# 63 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2




# 1 "/usr/include/string.h" 1 3 4
# 27 "/usr/include/string.h" 3 4
extern "C" {




# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 33 "/usr/include/string.h" 2 3 4









extern void *memcpy (void *__restrict __dest, const void *__restrict __src,
       size_t __n) throw () __attribute__ ((__nonnull__ (1, 2)));


extern void *memmove (void *__dest, const void *__src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));






extern void *memccpy (void *__restrict __dest, const void *__restrict __src,
        int __c, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));





extern void *memset (void *__s, int __c, size_t __n) throw () __attribute__ ((__nonnull__ (1)));


extern int memcmp (const void *__s1, const void *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));



extern "C++"
{
extern void *memchr (void *__s, int __c, size_t __n)
      throw () __asm ("memchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const void *memchr (const void *__s, int __c, size_t __n)
      throw () __asm ("memchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 90 "/usr/include/string.h" 3 4
}










extern "C++" void *rawmemchr (void *__s, int __c)
     throw () __asm ("rawmemchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern "C++" const void *rawmemchr (const void *__s, int __c)
     throw () __asm ("rawmemchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));







extern "C++" void *memrchr (void *__s, int __c, size_t __n)
      throw () __asm ("memrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern "C++" const void *memrchr (const void *__s, int __c, size_t __n)
      throw () __asm ("memrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));









extern char *strcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));

extern char *strncpy (char *__restrict __dest,
        const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern char *strcat (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));

extern char *strncat (char *__restrict __dest, const char *__restrict __src,
        size_t __n) throw () __attribute__ ((__nonnull__ (1, 2)));


extern int strcmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern int strncmp (const char *__s1, const char *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcoll (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern size_t strxfrm (char *__restrict __dest,
         const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (2)));






# 1 "/usr/include/xlocale.h" 1 3 4
# 27 "/usr/include/xlocale.h" 3 4
typedef struct __locale_struct
{

  struct __locale_data *__locales[13];


  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;


  const char *__names[13];
} *__locale_t;


typedef __locale_t locale_t;
# 160 "/usr/include/string.h" 2 3 4


extern int strcoll_l (const char *__s1, const char *__s2, __locale_t __l)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));

extern size_t strxfrm_l (char *__dest, const char *__src, size_t __n,
    __locale_t __l) throw () __attribute__ ((__nonnull__ (2, 4)));





extern char *strdup (const char *__s)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));






extern char *strndup (const char *__string, size_t __n)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));
# 207 "/usr/include/string.h" 3 4



extern "C++"
{
extern char *strchr (char *__s, int __c)
     throw () __asm ("strchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *strchr (const char *__s, int __c)
     throw () __asm ("strchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 230 "/usr/include/string.h" 3 4
}






extern "C++"
{
extern char *strrchr (char *__s, int __c)
     throw () __asm ("strrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *strrchr (const char *__s, int __c)
     throw () __asm ("strrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 257 "/usr/include/string.h" 3 4
}










extern "C++" char *strchrnul (char *__s, int __c)
     throw () __asm ("strchrnul") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern "C++" const char *strchrnul (const char *__s, int __c)
     throw () __asm ("strchrnul") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));









extern size_t strcspn (const char *__s, const char *__reject)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern size_t strspn (const char *__s, const char *__accept)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern "C++"
{
extern char *strpbrk (char *__s, const char *__accept)
     throw () __asm ("strpbrk") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern const char *strpbrk (const char *__s, const char *__accept)
     throw () __asm ("strpbrk") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 309 "/usr/include/string.h" 3 4
}






extern "C++"
{
extern char *strstr (char *__haystack, const char *__needle)
     throw () __asm ("strstr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern const char *strstr (const char *__haystack, const char *__needle)
     throw () __asm ("strstr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 336 "/usr/include/string.h" 3 4
}







extern char *strtok (char *__restrict __s, const char *__restrict __delim)
     throw () __attribute__ ((__nonnull__ (2)));




extern char *__strtok_r (char *__restrict __s,
    const char *__restrict __delim,
    char **__restrict __save_ptr)
     throw () __attribute__ ((__nonnull__ (2, 3)));

extern char *strtok_r (char *__restrict __s, const char *__restrict __delim,
         char **__restrict __save_ptr)
     throw () __attribute__ ((__nonnull__ (2, 3)));





extern "C++" char *strcasestr (char *__haystack, const char *__needle)
     throw () __asm ("strcasestr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern "C++" const char *strcasestr (const char *__haystack,
         const char *__needle)
     throw () __asm ("strcasestr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 378 "/usr/include/string.h" 3 4
extern void *memmem (const void *__haystack, size_t __haystacklen,
       const void *__needle, size_t __needlelen)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 3)));



extern void *__mempcpy (void *__restrict __dest,
   const void *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern void *mempcpy (void *__restrict __dest,
        const void *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));





extern size_t strlen (const char *__s)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));





extern size_t strnlen (const char *__string, size_t __maxlen)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));





extern char *strerror (int __errnum) throw ();

# 434 "/usr/include/string.h" 3 4
extern char *strerror_r (int __errnum, char *__buf, size_t __buflen)
     throw () __attribute__ ((__nonnull__ (2))) ;





extern char *strerror_l (int __errnum, __locale_t __l) throw ();





extern void __bzero (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)));



extern void bcopy (const void *__src, void *__dest, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern void bzero (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)));


extern int bcmp (const void *__s1, const void *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));



extern "C++"
{
extern char *index (char *__s, int __c)
     throw () __asm ("index") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *index (const char *__s, int __c)
     throw () __asm ("index") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 483 "/usr/include/string.h" 3 4
}







extern "C++"
{
extern char *rindex (char *__s, int __c)
     throw () __asm ("rindex") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *rindex (const char *__s, int __c)
     throw () __asm ("rindex") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 511 "/usr/include/string.h" 3 4
}







extern int ffs (int __i) throw () __attribute__ ((__const__));




extern int ffsl (long int __l) throw () __attribute__ ((__const__));

__extension__ extern int ffsll (long long int __ll)
     throw () __attribute__ ((__const__));




extern int strcasecmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (const char *__s1, const char *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));





extern int strcasecmp_l (const char *__s1, const char *__s2,
    __locale_t __loc)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));

extern int strncasecmp_l (const char *__s1, const char *__s2,
     size_t __n, __locale_t __loc)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 4)));





extern char *strsep (char **__restrict __stringp,
       const char *__restrict __delim)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern char *strsignal (int __sig) throw ();


extern char *__stpcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern char *stpcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));



extern char *__stpncpy (char *__restrict __dest,
   const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern char *stpncpy (char *__restrict __dest,
        const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern int strverscmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *strfry (char *__string) throw () __attribute__ ((__nonnull__ (1)));


extern void *memfrob (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)));







extern "C++" char *basename (char *__filename)
     throw () __asm ("basename") __attribute__ ((__nonnull__ (1)));
extern "C++" const char *basename (const char *__filename)
     throw () __asm ("basename") __attribute__ ((__nonnull__ (1)));
# 642 "/usr/include/string.h" 3 4
}
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2
# 1 "/usr/include/time.h" 1 3 4
# 29 "/usr/include/time.h" 3 4
extern "C" {







# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 38 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/time.h" 1 3 4
# 26 "/usr/include/bits/time.h" 3 4
# 1 "/usr/include/bits/types.h" 1 3 4
# 27 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/types.h" 2 3 4


typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;


typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;

typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;







typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
# 130 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/typesizes.h" 1 3 4
# 131 "/usr/include/bits/types.h" 2 3 4


typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;

typedef int __daddr_t;
typedef int __key_t;


typedef int __clockid_t;


typedef void * __timer_t;


typedef long int __blksize_t;




typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;


typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;


typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;


typedef long int __fsword_t;

typedef long int __ssize_t;


typedef long int __syscall_slong_t;

typedef unsigned long int __syscall_ulong_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;
# 27 "/usr/include/bits/time.h" 2 3 4



struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };
# 88 "/usr/include/bits/time.h" 3 4
# 1 "/usr/include/bits/timex.h" 1 3 4
# 25 "/usr/include/bits/timex.h" 3 4
struct timex
{
  unsigned int modes;
  __syscall_slong_t offset;
  __syscall_slong_t freq;
  __syscall_slong_t maxerror;
  __syscall_slong_t esterror;
  int status;
  __syscall_slong_t constant;
  __syscall_slong_t precision;
  __syscall_slong_t tolerance;
  struct timeval time;
  __syscall_slong_t tick;
  __syscall_slong_t ppsfreq;
  __syscall_slong_t jitter;
  int shift;
  __syscall_slong_t stabil;
  __syscall_slong_t jitcnt;
  __syscall_slong_t calcnt;
  __syscall_slong_t errcnt;
  __syscall_slong_t stbcnt;

  int tai;


  int :32; int :32; int :32; int :32;
  int :32; int :32; int :32; int :32;
  int :32; int :32; int :32;
};
# 89 "/usr/include/bits/time.h" 2 3 4

extern "C" {


extern int clock_adjtime (__clockid_t __clock_id, struct timex *__utx) throw ();

}
# 42 "/usr/include/time.h" 2 3 4
# 57 "/usr/include/time.h" 3 4


typedef __clock_t clock_t;



# 73 "/usr/include/time.h" 3 4


typedef __time_t time_t;



# 91 "/usr/include/time.h" 3 4
typedef __clockid_t clockid_t;
# 103 "/usr/include/time.h" 3 4
typedef __timer_t timer_t;
# 120 "/usr/include/time.h" 3 4
struct timespec
  {
    __time_t tv_sec;
    __syscall_slong_t tv_nsec;
  };








struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;


  long int tm_gmtoff;
  const char *tm_zone;




};








struct itimerspec
  {
    struct timespec it_interval;
    struct timespec it_value;
  };


struct sigevent;





typedef __pid_t pid_t;
# 186 "/usr/include/time.h" 3 4



extern clock_t clock (void) throw ();


extern time_t time (time_t *__timer) throw ();


extern double difftime (time_t __time1, time_t __time0)
     throw () __attribute__ ((__const__));


extern time_t mktime (struct tm *__tp) throw ();





extern size_t strftime (char *__restrict __s, size_t __maxsize,
   const char *__restrict __format,
   const struct tm *__restrict __tp) throw ();





extern char *strptime (const char *__restrict __s,
         const char *__restrict __fmt, struct tm *__tp)
     throw ();







extern size_t strftime_l (char *__restrict __s, size_t __maxsize,
     const char *__restrict __format,
     const struct tm *__restrict __tp,
     __locale_t __loc) throw ();



extern char *strptime_l (const char *__restrict __s,
    const char *__restrict __fmt, struct tm *__tp,
    __locale_t __loc) throw ();






extern struct tm *gmtime (const time_t *__timer) throw ();



extern struct tm *localtime (const time_t *__timer) throw ();





extern struct tm *gmtime_r (const time_t *__restrict __timer,
       struct tm *__restrict __tp) throw ();



extern struct tm *localtime_r (const time_t *__restrict __timer,
          struct tm *__restrict __tp) throw ();





extern char *asctime (const struct tm *__tp) throw ();


extern char *ctime (const time_t *__timer) throw ();







extern char *asctime_r (const struct tm *__restrict __tp,
   char *__restrict __buf) throw ();


extern char *ctime_r (const time_t *__restrict __timer,
        char *__restrict __buf) throw ();




extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;




extern char *tzname[2];



extern void tzset (void) throw ();



extern int daylight;
extern long int timezone;





extern int stime (const time_t *__when) throw ();
# 319 "/usr/include/time.h" 3 4
extern time_t timegm (struct tm *__tp) throw ();


extern time_t timelocal (struct tm *__tp) throw ();


extern int dysize (int __year) throw () __attribute__ ((__const__));
# 334 "/usr/include/time.h" 3 4
extern int nanosleep (const struct timespec *__requested_time,
        struct timespec *__remaining);



extern int clock_getres (clockid_t __clock_id, struct timespec *__res) throw ();


extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) throw ();


extern int clock_settime (clockid_t __clock_id, const struct timespec *__tp)
     throw ();






extern int clock_nanosleep (clockid_t __clock_id, int __flags,
       const struct timespec *__req,
       struct timespec *__rem);


extern int clock_getcpuclockid (pid_t __pid, clockid_t *__clock_id) throw ();




extern int timer_create (clockid_t __clock_id,
    struct sigevent *__restrict __evp,
    timer_t *__restrict __timerid) throw ();


extern int timer_delete (timer_t __timerid) throw ();


extern int timer_settime (timer_t __timerid, int __flags,
     const struct itimerspec *__restrict __value,
     struct itimerspec *__restrict __ovalue) throw ();


extern int timer_gettime (timer_t __timerid, struct itimerspec *__value)
     throw ();


extern int timer_getoverrun (timer_t __timerid) throw ();





extern int timespec_get (struct timespec *__ts, int __base)
     throw () __attribute__ ((__nonnull__ (1)));
# 403 "/usr/include/time.h" 3 4
extern int getdate_err;
# 412 "/usr/include/time.h" 3 4
extern struct tm *getdate (const char *__string);
# 426 "/usr/include/time.h" 3 4
extern int getdate_r (const char *__restrict __string,
        struct tm *__restrict __resbufp);


}
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2

extern "C"
{

extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) clock_t clock(void)



throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void* memset(void*, int, size_t) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void* memcpy(void*, const void*, size_t) throw ();

}
# 93 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 1 3
# 37 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 3
       
# 38 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 3

# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 1 3
# 186 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 3
namespace std
{
  typedef long unsigned int size_t;
  typedef long int ptrdiff_t;




}
# 350 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 3
namespace std
{
  inline namespace __gnu_cxx_ldbl128 { }
}
# 430 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 3
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/os_defines.h" 1 3
# 431 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 2 3


# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/cpu_defines.h" 1 3
# 434 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/powerpc64le-unknown-linux-gnu/bits/c++config.h" 2 3
# 40 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 2 3
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 1 3
# 33 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
       
# 34 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3

#pragma GCC visibility push(default)


# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/atomic_lockfree_defines.h" 1 3
# 33 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/atomic_lockfree_defines.h" 3
       
# 34 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/atomic_lockfree_defines.h" 3
# 39 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 2 3

extern "C++" {

namespace std
{
# 60 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
  class exception
  {
  public:
    exception() throw() { }
    virtual ~exception() throw();



    virtual const char* what() const throw();
  };



  class bad_exception : public exception
  {
  public:
    bad_exception() throw() { }



    virtual ~bad_exception() throw();


    virtual const char* what() const throw();
  };


  typedef void (*terminate_handler) ();


  typedef void (*unexpected_handler) ();


  terminate_handler set_terminate(terminate_handler) throw();
# 102 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
  void terminate() throw() __attribute__ ((__noreturn__));


  unexpected_handler set_unexpected(unexpected_handler) throw();
# 114 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
  void unexpected() __attribute__ ((__noreturn__));
# 127 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
  bool uncaught_exception() throw() __attribute__ ((__pure__));


}

namespace __gnu_cxx
{

# 152 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/exception" 3
  void __verbose_terminate_handler();


}

}

#pragma GCC visibility pop
# 41 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 2 3

#pragma GCC visibility push(default)

extern "C++" {

namespace std
{






  class bad_alloc : public exception
  {
  public:
    bad_alloc() throw() { }



    virtual ~bad_alloc() throw();


    virtual const char* what() const throw();
  };
# 85 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 3
  class bad_array_length : public bad_alloc
  {
  public:
    bad_array_length() throw() { };



    virtual ~bad_array_length() throw();


    virtual const char* what() const throw();
  };


  struct nothrow_t { };

  extern const nothrow_t nothrow;



  typedef void (*new_handler)();



  new_handler set_new_handler(new_handler) throw();





}
# 128 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/new" 3
void* operator new(std::size_t) throw(std::bad_alloc)
  __attribute__((__externally_visible__));
void* operator new[](std::size_t) throw(std::bad_alloc)
  __attribute__((__externally_visible__));
void operator delete(void*) throw()
  __attribute__((__externally_visible__));
void operator delete[](void*) throw()
  __attribute__((__externally_visible__));
void* operator new(std::size_t, const std::nothrow_t&) throw()
  __attribute__((__externally_visible__));
void* operator new[](std::size_t, const std::nothrow_t&) throw()
  __attribute__((__externally_visible__));
void operator delete(void*, const std::nothrow_t&) throw()
  __attribute__((__externally_visible__));
void operator delete[](void*, const std::nothrow_t&) throw()
  __attribute__((__externally_visible__));


inline void* operator new(std::size_t, void* __p) throw()
{ return __p; }
inline void* operator new[](std::size_t, void* __p) throw()
{ return __p; }


inline void operator delete (void*, void*) throw() { }
inline void operator delete[](void*, void*) throw() { }

}

#pragma GCC visibility pop
# 94 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2
# 107 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new(std:: size_t, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new[](std:: size_t, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*, void*) throw();







# 1 "/usr/include/stdio.h" 1 3 4
# 29 "/usr/include/stdio.h" 3 4
extern "C" {



# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 34 "/usr/include/stdio.h" 2 3 4
# 44 "/usr/include/stdio.h" 3 4
struct _IO_FILE;



typedef struct _IO_FILE FILE;





# 64 "/usr/include/stdio.h" 3 4
typedef struct _IO_FILE __FILE;
# 74 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/libio.h" 1 3 4
# 32 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_G_config.h" 1 3 4
# 15 "/usr/include/_G_config.h" 3 4
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 16 "/usr/include/_G_config.h" 2 3 4




# 1 "/usr/include/wchar.h" 1 3 4
# 82 "/usr/include/wchar.h" 3 4
typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;
# 21 "/usr/include/_G_config.h" 2 3 4
typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
# 33 "/usr/include/libio.h" 2 3 4
# 50 "/usr/include/libio.h" 3 4
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stdarg.h" 1 3 4
# 40 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 51 "/usr/include/libio.h" 2 3 4
# 145 "/usr/include/libio.h" 3 4
struct _IO_jump_t; struct _IO_FILE;
# 155 "/usr/include/libio.h" 3 4
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
# 178 "/usr/include/libio.h" 3 4
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
# 246 "/usr/include/libio.h" 3 4
struct _IO_FILE {
  int _flags;




  char* _IO_read_ptr;
  char* _IO_read_end;
  char* _IO_read_base;
  char* _IO_write_base;
  char* _IO_write_ptr;
  char* _IO_write_end;
  char* _IO_buf_base;
  char* _IO_buf_end;

  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;



  int _flags2;

  __off_t _old_offset;



  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];



  _IO_lock_t *_lock;
# 294 "/usr/include/libio.h" 3 4
  __off64_t _offset;
# 303 "/usr/include/libio.h" 3 4
  void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;
  size_t __pad5;

  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];

};





struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
# 339 "/usr/include/libio.h" 3 4
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);




typedef __io_read_fn cookie_read_function_t;
typedef __io_write_fn cookie_write_function_t;
typedef __io_seek_fn cookie_seek_function_t;
typedef __io_close_fn cookie_close_function_t;


typedef struct
{
  __io_read_fn *read;
  __io_write_fn *write;
  __io_seek_fn *seek;
  __io_close_fn *close;
} _IO_cookie_io_functions_t;
typedef _IO_cookie_io_functions_t cookie_io_functions_t;

struct _IO_cookie_file;


extern void _IO_cookie_init (struct _IO_cookie_file *__cfile, int __read_write,
        void *__cookie, _IO_cookie_io_functions_t __fns);




extern "C" {


extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
# 435 "/usr/include/libio.h" 3 4
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) throw ();
extern int _IO_ferror (_IO_FILE *__fp) throw ();

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) throw ();
extern void _IO_funlockfile (_IO_FILE *) throw ();
extern int _IO_ftrylockfile (_IO_FILE *) throw ();
# 465 "/usr/include/libio.h" 3 4
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) throw ();
# 527 "/usr/include/libio.h" 3 4
}
# 75 "/usr/include/stdio.h" 2 3 4




typedef __gnuc_va_list va_list;
# 90 "/usr/include/stdio.h" 3 4
typedef __off_t off_t;






typedef __off64_t off64_t;




typedef __ssize_t ssize_t;







typedef _G_fpos_t fpos_t;





typedef _G_fpos64_t fpos64_t;
# 164 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 165 "/usr/include/stdio.h" 2 3 4



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;







extern int remove (const char *__filename) throw ();

extern int rename (const char *__old, const char *__new) throw ();




extern int renameat (int __oldfd, const char *__old, int __newfd,
       const char *__new) throw ();








extern FILE *tmpfile (void) ;
# 205 "/usr/include/stdio.h" 3 4
extern FILE *tmpfile64 (void) ;



extern char *tmpnam (char *__s) throw () ;





extern char *tmpnam_r (char *__s) throw () ;
# 227 "/usr/include/stdio.h" 3 4
extern char *tempnam (const char *__dir, const char *__pfx)
     throw () __attribute__ ((__malloc__)) ;








extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);

# 252 "/usr/include/stdio.h" 3 4
extern int fflush_unlocked (FILE *__stream);
# 262 "/usr/include/stdio.h" 3 4
extern int fcloseall (void);









extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) ;




extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 295 "/usr/include/stdio.h" 3 4


extern FILE *fopen64 (const char *__restrict __filename,
        const char *__restrict __modes) ;
extern FILE *freopen64 (const char *__restrict __filename,
   const char *__restrict __modes,
   FILE *__restrict __stream) ;




extern FILE *fdopen (int __fd, const char *__modes) throw () ;





extern FILE *fopencookie (void *__restrict __magic_cookie,
     const char *__restrict __modes,
     _IO_cookie_io_functions_t __io_funcs) throw () ;




extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  throw () ;




extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) throw () ;






extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) throw ();



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) throw ();





extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) throw ();


extern void setlinebuf (FILE *__stream) throw ();








extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);




extern int printf (const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) throw ();





extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) throw ();





extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     throw () __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__printf__, 3, 0)));






extern int vasprintf (char **__restrict __ptr, const char *__restrict __f,
        __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__printf__, 2, 0))) ;
extern int __asprintf (char **__restrict __ptr,
         const char *__restrict __fmt, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3))) ;
extern int asprintf (char **__restrict __ptr,
       const char *__restrict __fmt, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3))) ;




extern int vdprintf (int __fd, const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));








extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) ;




extern int scanf (const char *__restrict __format, ...) ;

extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) throw ();
# 463 "/usr/include/stdio.h" 3 4








extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;





extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;


extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__scanf__, 2, 0)));
# 522 "/usr/include/stdio.h" 3 4









extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);

# 550 "/usr/include/stdio.h" 3 4
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 561 "/usr/include/stdio.h" 3 4
extern int fgetc_unlocked (FILE *__stream);











extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);

# 594 "/usr/include/stdio.h" 3 4
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);








extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;
# 638 "/usr/include/stdio.h" 3 4
extern char *gets (char *__s) __attribute__ ((__deprecated__));


# 649 "/usr/include/stdio.h" 3 4
extern char *fgets_unlocked (char *__restrict __s, int __n,
        FILE *__restrict __stream) ;
# 665 "/usr/include/stdio.h" 3 4
extern __ssize_t __getdelim (char **__restrict __lineptr,
          size_t *__restrict __n, int __delimiter,
          FILE *__restrict __stream) ;
extern __ssize_t getdelim (char **__restrict __lineptr,
        size_t *__restrict __n, int __delimiter,
        FILE *__restrict __stream) ;







extern __ssize_t getline (char **__restrict __lineptr,
       size_t *__restrict __n,
       FILE *__restrict __stream) ;








extern int fputs (const char *__restrict __s, FILE *__restrict __stream);





extern int puts (const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);

# 726 "/usr/include/stdio.h" 3 4
extern int fputs_unlocked (const char *__restrict __s,
      FILE *__restrict __stream);
# 737 "/usr/include/stdio.h" 3 4
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream);








extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);

# 773 "/usr/include/stdio.h" 3 4
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 792 "/usr/include/stdio.h" 3 4






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, const fpos_t *__pos);
# 815 "/usr/include/stdio.h" 3 4



extern int fseeko64 (FILE *__stream, __off64_t __off, int __whence);
extern __off64_t ftello64 (FILE *__stream) ;
extern int fgetpos64 (FILE *__restrict __stream, fpos64_t *__restrict __pos);
extern int fsetpos64 (FILE *__stream, const fpos64_t *__pos);




extern void clearerr (FILE *__stream) throw ();

extern int feof (FILE *__stream) throw () ;

extern int ferror (FILE *__stream) throw () ;




extern void clearerr_unlocked (FILE *__stream) throw ();
extern int feof_unlocked (FILE *__stream) throw () ;
extern int ferror_unlocked (FILE *__stream) throw () ;








extern void perror (const char *__s);






# 1 "/usr/include/bits/sys_errlist.h" 1 3 4
# 26 "/usr/include/bits/sys_errlist.h" 3 4
extern int sys_nerr;
extern const char *const sys_errlist[];


extern int _sys_nerr;
extern const char *const _sys_errlist[];
# 854 "/usr/include/stdio.h" 2 3 4




extern int fileno (FILE *__stream) throw () ;




extern int fileno_unlocked (FILE *__stream) throw () ;
# 873 "/usr/include/stdio.h" 3 4
extern FILE *popen (const char *__command, const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) throw ();





extern char *cuserid (char *__s);




struct obstack;


extern int obstack_printf (struct obstack *__restrict __obstack,
      const char *__restrict __format, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3)));
extern int obstack_vprintf (struct obstack *__restrict __obstack,
       const char *__restrict __format,
       __gnuc_va_list __args)
     throw () __attribute__ ((__format__ (__printf__, 2, 0)));







extern void flockfile (FILE *__stream) throw ();



extern int ftrylockfile (FILE *__stream) throw () ;


extern void funlockfile (FILE *__stream) throw ();
# 943 "/usr/include/stdio.h" 3 4
}
# 119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2
# 1 "/usr/include/stdlib.h" 1 3 4
# 32 "/usr/include/stdlib.h" 3 4
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 33 "/usr/include/stdlib.h" 2 3 4

extern "C" {






# 1 "/usr/include/bits/waitflags.h" 1 3 4
# 42 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/bits/waitstatus.h" 1 3 4
# 64 "/usr/include/bits/waitstatus.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
# 36 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/endian.h" 1 3 4
# 37 "/usr/include/endian.h" 2 3 4
# 60 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/byteswap.h" 1 3 4
# 34 "/usr/include/bits/byteswap.h" 3 4
# 1 "/usr/include/bits/byteswap-16.h" 1 3 4
# 35 "/usr/include/bits/byteswap.h" 2 3 4
# 43 "/usr/include/bits/byteswap.h" 3 4
static __inline unsigned int
__bswap_32 (unsigned int __bsx)
{
  return __builtin_bswap32 (__bsx);
}
# 74 "/usr/include/bits/byteswap.h" 3 4
static __inline __uint64_t
__bswap_64 (__uint64_t __bsx)
{
  return __builtin_bswap64 (__bsx);
}
# 61 "/usr/include/endian.h" 2 3 4
# 65 "/usr/include/bits/waitstatus.h" 2 3 4

union wait
  {
    int w_status;
    struct
      {

 unsigned int __w_termsig:7;
 unsigned int __w_coredump:1;
 unsigned int __w_retcode:8;
 unsigned int:16;







      } __wait_terminated;
    struct
      {

 unsigned int __w_stopval:8;
 unsigned int __w_stopsig:8;
 unsigned int:16;






      } __wait_stopped;
  };
# 43 "/usr/include/stdlib.h" 2 3 4
# 95 "/usr/include/stdlib.h" 3 4


typedef struct
  {
    int quot;
    int rem;
  } div_t;



typedef struct
  {
    long int quot;
    long int rem;
  } ldiv_t;







__extension__ typedef struct
  {
    long long int quot;
    long long int rem;
  } lldiv_t;


# 139 "/usr/include/stdlib.h" 3 4
extern size_t __ctype_get_mb_cur_max (void) throw () ;




extern double atof (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





__extension__ extern long long int atoll (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));





extern float strtof (const char *__restrict __nptr,
       char **__restrict __endptr) throw () __attribute__ ((__nonnull__ (1)));

extern long double strtold (const char *__restrict __nptr,
       char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));





extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));




__extension__
extern long long int strtoq (const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));





__extension__
extern long long int strtoll (const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

# 239 "/usr/include/stdlib.h" 3 4
extern long int strtol_l (const char *__restrict __nptr,
     char **__restrict __endptr, int __base,
     __locale_t __loc) throw () __attribute__ ((__nonnull__ (1, 4)));

extern unsigned long int strtoul_l (const char *__restrict __nptr,
        char **__restrict __endptr,
        int __base, __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

__extension__
extern long long int strtoll_l (const char *__restrict __nptr,
    char **__restrict __endptr, int __base,
    __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

__extension__
extern unsigned long long int strtoull_l (const char *__restrict __nptr,
       char **__restrict __endptr,
       int __base, __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

extern double strtod_l (const char *__restrict __nptr,
   char **__restrict __endptr, __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));

extern float strtof_l (const char *__restrict __nptr,
         char **__restrict __endptr, __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));

extern long double strtold_l (const char *__restrict __nptr,
         char **__restrict __endptr,
         __locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));
# 305 "/usr/include/stdlib.h" 3 4
extern char *l64a (long int __n) throw () ;


extern long int a64l (const char *__s)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;




# 1 "/usr/include/sys/types.h" 1 3 4
# 27 "/usr/include/sys/types.h" 3 4
extern "C" {





typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;




typedef __loff_t loff_t;



typedef __ino_t ino_t;






typedef __ino64_t ino64_t;




typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;
# 104 "/usr/include/sys/types.h" 3 4
typedef __id_t id_t;
# 115 "/usr/include/sys/types.h" 3 4
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 136 "/usr/include/sys/types.h" 3 4
typedef __useconds_t useconds_t;



typedef __suseconds_t suseconds_t;





# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 147 "/usr/include/sys/types.h" 2 3 4



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
# 194 "/usr/include/sys/types.h" 3 4
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));
# 219 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/sys/select.h" 1 3 4
# 30 "/usr/include/sys/select.h" 3 4
# 1 "/usr/include/bits/select.h" 1 3 4
# 31 "/usr/include/sys/select.h" 2 3 4


# 1 "/usr/include/bits/sigset.h" 1 3 4
# 23 "/usr/include/bits/sigset.h" 3 4
typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;
# 34 "/usr/include/sys/select.h" 2 3 4



typedef __sigset_t sigset_t;







# 1 "/usr/include/bits/time.h" 1 3 4
# 46 "/usr/include/sys/select.h" 2 3 4
# 54 "/usr/include/sys/select.h" 3 4
typedef long int __fd_mask;
# 64 "/usr/include/sys/select.h" 3 4
typedef struct
  {



    __fd_mask fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];





  } fd_set;






typedef __fd_mask fd_mask;
# 96 "/usr/include/sys/select.h" 3 4
extern "C" {
# 106 "/usr/include/sys/select.h" 3 4
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 118 "/usr/include/sys/select.h" 3 4
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);
# 131 "/usr/include/sys/select.h" 3 4
}
# 220 "/usr/include/sys/types.h" 2 3 4


# 1 "/usr/include/sys/sysmacros.h" 1 3 4
# 29 "/usr/include/sys/sysmacros.h" 3 4
extern "C" {

__extension__
extern unsigned int gnu_dev_major (unsigned long long int __dev)
     throw () __attribute__ ((__const__));
__extension__
extern unsigned int gnu_dev_minor (unsigned long long int __dev)
     throw () __attribute__ ((__const__));
__extension__
extern unsigned long long int gnu_dev_makedev (unsigned int __major,
            unsigned int __minor)
     throw () __attribute__ ((__const__));
# 63 "/usr/include/sys/sysmacros.h" 3 4
}
# 223 "/usr/include/sys/types.h" 2 3 4





typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 262 "/usr/include/sys/types.h" 3 4
typedef __blkcnt64_t blkcnt64_t;
typedef __fsblkcnt64_t fsblkcnt64_t;
typedef __fsfilcnt64_t fsfilcnt64_t;





# 1 "/usr/include/bits/pthreadtypes.h" 1 3 4
# 22 "/usr/include/bits/pthreadtypes.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 23 "/usr/include/bits/pthreadtypes.h" 2 3 4
# 49 "/usr/include/bits/pthreadtypes.h" 3 4
typedef unsigned long int pthread_t;


union pthread_attr_t
{
  char __size[56];
  long int __align;
};

typedef union pthread_attr_t pthread_attr_t;





typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
# 79 "/usr/include/bits/pthreadtypes.h" 3 4
typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;

    unsigned int __nusers;



    int __kind;

    short __spins;
    short __elision;
    __pthread_list_t __list;
# 113 "/usr/include/bits/pthreadtypes.h" 3 4
  } __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_mutexattr_t;




typedef union
{
  struct
  {
    int __lock;
    unsigned int __futex;
    __extension__ unsigned long long int __total_seq;
    __extension__ unsigned long long int __wakeup_seq;
    __extension__ unsigned long long int __woken_seq;
    void *__mutex;
    unsigned int __nwaiters;
    unsigned int __broadcast_seq;
  } __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_condattr_t;



typedef unsigned int pthread_key_t;



typedef int pthread_once_t;





typedef union
{

  struct
  {
    int __lock;
    unsigned int __nr_readers;
    unsigned int __readers_wakeup;
    unsigned int __writer_wakeup;
    unsigned int __nr_readers_queued;
    unsigned int __nr_writers_queued;
    int __writer;
    int __shared;
    unsigned long int __pad1;
    unsigned long int __pad2;


    unsigned int __flags;
  } __data;
# 199 "/usr/include/bits/pthreadtypes.h" 3 4
  char __size[56];
  long int __align;
} pthread_rwlock_t;

typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;





typedef volatile int pthread_spinlock_t;




typedef union
{
  char __size[32];
  long int __align;
} pthread_barrier_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;
# 271 "/usr/include/sys/types.h" 2 3 4


}
# 315 "/usr/include/stdlib.h" 2 3 4






extern long int random (void) throw ();


extern void srandom (unsigned int __seed) throw ();





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) throw () __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) throw () __attribute__ ((__nonnull__ (1)));







struct random_data
  {
    int32_t *fptr;
    int32_t *rptr;
    int32_t *state;
    int rand_type;
    int rand_deg;
    int rand_sep;
    int32_t *end_ptr;
  };

extern int random_r (struct random_data *__restrict __buf,
       int32_t *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     throw () __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     throw () __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     throw () __attribute__ ((__nonnull__ (1, 2)));






extern int rand (void) throw ();

extern void srand (unsigned int __seed) throw ();




extern int rand_r (unsigned int *__seed) throw ();







extern double drand48 (void) throw ();
extern double erand48 (unsigned short int __xsubi[3]) throw () __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) throw ();
extern long int nrand48 (unsigned short int __xsubi[3])
     throw () __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) throw ();
extern long int jrand48 (unsigned short int __xsubi[3])
     throw () __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) throw ();
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     throw () __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) throw () __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    unsigned long long int __a;
  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     throw () __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) throw () __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     throw () __attribute__ ((__nonnull__ (1, 2)));









extern void *malloc (size_t __size) throw () __attribute__ ((__malloc__)) ;

extern void *calloc (size_t __nmemb, size_t __size)
     throw () __attribute__ ((__malloc__)) ;










extern void *realloc (void *__ptr, size_t __size)
     throw () __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) throw ();




extern void cfree (void *__ptr) throw ();



# 1 "/usr/include/alloca.h" 1 3 4
# 24 "/usr/include/alloca.h" 3 4
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3/include/stddef.h" 1 3 4
# 25 "/usr/include/alloca.h" 2 3 4

extern "C" {





extern void *alloca (size_t __size) throw ();





}
# 492 "/usr/include/stdlib.h" 2 3 4





extern void *valloc (size_t __size) throw () __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     throw () __attribute__ ((__nonnull__ (1))) ;




extern void *aligned_alloc (size_t __alignment, size_t __size)
     throw () __attribute__ ((__malloc__, __alloc_size__ (2)));




extern void abort (void) throw () __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) throw () __attribute__ ((__nonnull__ (1)));




extern "C++" int at_quick_exit (void (*__func) (void))
     throw () __asm ("at_quick_exit") __attribute__ ((__nonnull__ (1)));









extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     throw () __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) throw () __attribute__ ((__noreturn__));





extern void quick_exit (int __status) throw () __attribute__ ((__noreturn__));







extern void _Exit (int __status) throw () __attribute__ ((__noreturn__));






extern char *getenv (const char *__name) throw () __attribute__ ((__nonnull__ (1))) ;





extern char *secure_getenv (const char *__name)
     throw () __attribute__ ((__nonnull__ (1))) ;






extern int putenv (char *__string) throw () __attribute__ ((__nonnull__ (1)));





extern int setenv (const char *__name, const char *__value, int __replace)
     throw () __attribute__ ((__nonnull__ (2)));


extern int unsetenv (const char *__name) throw () __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) throw ();
# 605 "/usr/include/stdlib.h" 3 4
extern char *mktemp (char *__template) throw () __attribute__ ((__nonnull__ (1)));
# 619 "/usr/include/stdlib.h" 3 4
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 629 "/usr/include/stdlib.h" 3 4
extern int mkstemp64 (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 641 "/usr/include/stdlib.h" 3 4
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
# 651 "/usr/include/stdlib.h" 3 4
extern int mkstemps64 (char *__template, int __suffixlen)
     __attribute__ ((__nonnull__ (1))) ;
# 662 "/usr/include/stdlib.h" 3 4
extern char *mkdtemp (char *__template) throw () __attribute__ ((__nonnull__ (1))) ;
# 673 "/usr/include/stdlib.h" 3 4
extern int mkostemp (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 683 "/usr/include/stdlib.h" 3 4
extern int mkostemp64 (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 693 "/usr/include/stdlib.h" 3 4
extern int mkostemps (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;
# 705 "/usr/include/stdlib.h" 3 4
extern int mkostemps64 (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;









extern int system (const char *__command) ;






extern char *canonicalize_file_name (const char *__name)
     throw () __attribute__ ((__nonnull__ (1))) ;
# 733 "/usr/include/stdlib.h" 3 4
extern char *realpath (const char *__restrict __name,
         char *__restrict __resolved) throw () ;






typedef int (*__compar_fn_t) (const void *, const void *);


typedef __compar_fn_t comparison_fn_t;



typedef int (*__compar_d_fn_t) (const void *, const void *, void *);





extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;



extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));

extern void qsort_r (void *__base, size_t __nmemb, size_t __size,
       __compar_d_fn_t __compar, void *__arg)
  __attribute__ ((__nonnull__ (1, 4)));




extern int abs (int __x) throw () __attribute__ ((__const__)) ;
extern long int labs (long int __x) throw () __attribute__ ((__const__)) ;



__extension__ extern long long int llabs (long long int __x)
     throw () __attribute__ ((__const__)) ;







extern div_t div (int __numer, int __denom)
     throw () __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     throw () __attribute__ ((__const__)) ;




__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     throw () __attribute__ ((__const__)) ;

# 807 "/usr/include/stdlib.h" 3 4
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) throw () __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) throw () __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     throw () __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     throw () __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     throw () __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     throw () __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) throw () __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) throw () __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     throw () __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     throw () __attribute__ ((__nonnull__ (3, 4, 5)));







extern int mblen (const char *__s, size_t __n) throw () ;


extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n) throw () ;


extern int wctomb (char *__s, wchar_t __wchar) throw () ;



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n) throw ();

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n)
     throw ();








extern int rpmatch (const char *__response) throw () __attribute__ ((__nonnull__ (1))) ;
# 895 "/usr/include/stdlib.h" 3 4
extern int getsubopt (char **__restrict __optionp,
        char *const *__restrict __tokens,
        char **__restrict __valuep)
     throw () __attribute__ ((__nonnull__ (1, 2, 3))) ;





extern void setkey (const char *__key) throw () __attribute__ ((__nonnull__ (1)));







extern int posix_openpt (int __oflag) ;







extern int grantpt (int __fd) throw ();



extern int unlockpt (int __fd) throw ();




extern char *ptsname (int __fd) throw () ;






extern int ptsname_r (int __fd, char *__buf, size_t __buflen)
     throw () __attribute__ ((__nonnull__ (2)));


extern int getpt (void);






extern int getloadavg (double __loadavg[], int __nelem)
     throw () __attribute__ ((__nonnull__ (1)));


# 1 "/usr/include/bits/stdlib-float.h" 1 3 4
# 952 "/usr/include/stdlib.h" 2 3 4
# 964 "/usr/include/stdlib.h" 3 4
}
# 120 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2





extern "C"
{
extern







__attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) int printf(const char*, ...);



extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* malloc(size_t) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void free(void*) throw ();

}





# 1 "/usr/include/assert.h" 1 3 4
# 65 "/usr/include/assert.h" 3 4
extern "C" {


extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));


extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));




extern void __assert (const char *__assertion, const char *__file, int __line)
     throw () __attribute__ ((__noreturn__));


}
# 149 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2


extern "C"
{
# 179 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void __assert_fail(
  const char *, const char *, unsigned int, const char *)
  throw ();




}
# 230 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new(std:: size_t) throw(std:: bad_alloc);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new[](std:: size_t) throw(std:: bad_alloc);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*) throw();
# 257 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 1
# 89 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 90 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 2
# 98 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern "C"
{
# 182 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) int abs(int) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) long int labs(long int) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) long long int llabs(long long int) throw ();
# 234 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fabs(double x) throw ();
# 275 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fabsf(float x) throw ();



extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int min(int, int);

extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned int umin(unsigned int, unsigned int);
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llmin(long long int, long long int);
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned long long int ullmin(unsigned long long int, unsigned long long int);
# 304 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fminf(float x, float y) throw ();
# 324 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmin(double x, double y) throw ();






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int max(int, int);

extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned int umax(unsigned int, unsigned int);
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llmax(long long int, long long int);
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned long long int ullmax(unsigned long long int, unsigned long long int);
# 356 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmaxf(float x, float y) throw ();
# 376 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmax(double, double) throw ();
# 420 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sin(double x) throw ();
# 453 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cos(double x) throw ();
# 472 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincos(double x, double *sptr, double *cptr) throw ();
# 488 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincosf(float x, float *sptr, float *cptr) throw ();
# 533 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tan(double x) throw ();
# 602 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sqrt(double x) throw ();
# 674 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rsqrt(double x);
# 744 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rsqrtf(float x);
# 800 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log2(double x) throw ();
# 825 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp2(double x) throw ();
# 850 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float exp2f(float x) throw ();
# 877 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp10(double x) throw ();
# 900 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float exp10f(float x) throw ();
# 946 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double expm1(double x) throw ();
# 991 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float expm1f(float x) throw ();
# 1046 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log2f(float x) throw ();
# 1100 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log10(double x) throw ();
# 1171 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log(double x) throw ();
# 1265 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log1p(double x) throw ();
# 1362 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log1pf(float x) throw ();
# 1437 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double floor(double x) throw ();
# 1476 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp(double x) throw ();
# 1507 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cosh(double x) throw ();
# 1537 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sinh(double x) throw ();
# 1567 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tanh(double x) throw ();
# 1602 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double acosh(double x) throw ();
# 1640 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float acoshf(float x) throw ();
# 1656 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double asinh(double x) throw ();
# 1672 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float asinhf(float x) throw ();
# 1726 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atanh(double x) throw ();
# 1780 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atanhf(float x) throw ();
# 1839 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double ldexp(double x, int exp) throw ();
# 1895 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ldexpf(float x, int exp) throw ();
# 1947 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double logb(double x) throw ();
# 2002 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float logbf(float x) throw ();
# 2032 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int ilogb(double x) throw ();
# 2062 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int ilogbf(float x) throw ();
# 2138 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double scalbn(double x, int n) throw ();
# 2214 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float scalbnf(float x, int n) throw ();
# 2290 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double scalbln(double x, long int n) throw ();
# 2366 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float scalblnf(float x, long int n) throw ();
# 2444 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double frexp(double x, int *nptr) throw ();
# 2519 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float frexpf(float x, int *nptr) throw ();
# 2533 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double round(double x) throw ();
# 2550 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float roundf(float x) throw ();
# 2568 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lround(double x) throw ();
# 2586 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lroundf(float x) throw ();
# 2604 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llround(double x) throw ();
# 2622 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llroundf(float x) throw ();
# 2658 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rint(double x) throw ();
# 2674 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rintf(float x) throw ();
# 2691 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lrint(double x) throw ();
# 2708 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lrintf(float x) throw ();
# 2725 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llrint(double x) throw ();
# 2742 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llrintf(float x) throw ();
# 2795 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nearbyint(double x) throw ();
# 2848 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nearbyintf(float x) throw ();
# 2910 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double ceil(double x) throw ();
# 2922 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double trunc(double x) throw ();
# 2937 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float truncf(float x) throw ();
# 2963 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fdim(double x, double y) throw ();
# 2989 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fdimf(float x, float y) throw ();
# 3025 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atan2(double y, double x) throw ();
# 3056 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atan(double x) throw ();
# 3079 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double acos(double x) throw ();
# 3111 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double asin(double x) throw ();
# 3157 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double hypot(double x, double y) throw ();
# 3209 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rhypot(double x, double y) throw ();
# 3255 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float hypotf(float x, float y) throw ();
# 3307 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rhypotf(float x, float y) throw ();
# 3351 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double norm3d(double a, double b, double c) throw ();
# 3402 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rnorm3d(double a, double b, double c) throw ();
# 3451 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double norm4d(double a, double b, double c, double d) throw ();
# 3507 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rnorm4d(double a, double b, double c, double d) throw ();
# 3552 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double norm(int dim, double const * t) throw ();
# 3603 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rnorm(int dim, double const * t) throw ();
# 3655 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rnormf(int dim, float const * a) throw ();
# 3699 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float normf(int dim, float const * a) throw ();
# 3744 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float norm3df(float a, float b, float c) throw ();
# 3795 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rnorm3df(float a, float b, float c) throw ();
# 3844 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float norm4df(float a, float b, float c, float d) throw ();
# 3900 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rnorm4df(float a, float b, float c, float d) throw ();
# 3987 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cbrt(double x) throw ();
# 4073 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cbrtf(float x) throw ();
# 4128 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rcbrt(double x);
# 4178 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rcbrtf(float x);
# 4238 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sinpi(double x);
# 4298 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinpif(float x);
# 4350 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cospi(double x);
# 4402 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cospif(float x);
# 4432 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincospi(double x, double *sptr, double *cptr);
# 4462 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincospif(float x, float *sptr, float *cptr);
# 4774 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double pow(double x, double y) throw ();
# 4830 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double modf(double x, double *iptr) throw ();
# 4889 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmod(double x, double y) throw ();
# 4975 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double remainder(double x, double y) throw ();
# 5065 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float remainderf(float x, float y) throw ();
# 5119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double remquo(double x, double y, int *quo) throw ();
# 5173 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float remquof(float x, float y, int *quo) throw ();
# 5214 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double j0(double x) throw ();
# 5256 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float j0f(float x) throw ();
# 5317 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double j1(double x) throw ();
# 5378 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float j1f(float x) throw ();
# 5421 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double jn(int n, double x) throw ();
# 5464 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float jnf(int n, float x) throw ();
# 5516 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double y0(double x) throw ();
# 5568 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float y0f(float x) throw ();
# 5620 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double y1(double x) throw ();
# 5672 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float y1f(float x) throw ();
# 5725 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double yn(int n, double x) throw ();
# 5778 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ynf(int n, float x) throw ();
# 5805 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cyl_bessel_i0(double x) throw ();
# 5831 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cyl_bessel_i0f(float x) throw ();
# 5858 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cyl_bessel_i1(double x) throw ();
# 5884 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cyl_bessel_i1f(float x) throw ();
# 5967 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erf(double x) throw ();
# 6049 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erff(float x) throw ();
# 6113 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfinv(double y);
# 6170 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfinvf(float y);
# 6209 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfc(double x) throw ();
# 6247 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcf(float x) throw ();
# 6375 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double lgamma(double x) throw ();
# 6438 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfcinv(double y);
# 6494 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcinvf(float y);
# 6552 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double normcdfinv(double y);
# 6610 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float normcdfinvf(float y);
# 6653 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double normcdf(double y);
# 6696 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float normcdff(float y);
# 6771 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfcx(double x);
# 6846 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcxf(float x);
# 6980 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float lgammaf(float x) throw ();
# 7089 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tgamma(double x) throw ();
# 7198 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tgammaf(float x) throw ();
# 7211 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double copysign(double x, double y) throw ();
# 7224 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float copysignf(float x, float y) throw ();
# 7261 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nextafter(double x, double y) throw ();
# 7298 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nextafterf(float x, float y) throw ();
# 7314 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nan(const char *tagp) throw ();
# 7330 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nanf(const char *tagp) throw ();






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinff(float) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnanf(float) throw ();
# 7348 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finite(double) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finitef(float) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbit(double) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnan(double) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinf(double) throw ();


extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbitf(float) throw ();
# 7514 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fma(double x, double y, double z) throw ();
# 7672 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmaf(float x, float y, float z) throw ();
# 7683 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbitl(long double) throw ();





extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finitel(long double) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinfl(long double) throw ();
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnanl(long double) throw ();
# 7741 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float acosf(float x) throw ();
# 7781 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float asinf(float x) throw ();
# 7821 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atanf(float x) throw ();
# 7854 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atan2f(float y, float x) throw ();
# 7878 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cosf(float x) throw ();
# 7920 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinf(float x) throw ();
# 7962 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tanf(float x) throw ();
# 7986 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float coshf(float x) throw ();
# 8027 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinhf(float x) throw ();
# 8057 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tanhf(float x) throw ();
# 8108 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float logf(float x) throw ();
# 8158 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float expf(float x) throw ();
# 8209 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log10f(float x) throw ();
# 8264 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float modff(float x, float *iptr) throw ();
# 8572 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float powf(float x, float y) throw ();
# 8641 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sqrtf(float x) throw ();
# 8700 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ceilf(float x) throw ();
# 8772 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float floorf(float x) throw ();
# 8831 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmodf(float x, float y) throw ();
# 8846 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
}


# 1 "/usr/include/math.h" 1 3 4
# 29 "/usr/include/math.h" 3 4
extern "C" {



# 1 "/usr/include/bits/huge_val.h" 1 3 4
# 34 "/usr/include/math.h" 2 3 4

# 1 "/usr/include/bits/huge_valf.h" 1 3 4
# 36 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/huge_vall.h" 1 3 4
# 37 "/usr/include/math.h" 2 3 4


# 1 "/usr/include/bits/inf.h" 1 3 4
# 40 "/usr/include/math.h" 2 3 4


# 1 "/usr/include/bits/nan.h" 1 3 4
# 43 "/usr/include/math.h" 2 3 4



# 1 "/usr/include/bits/mathdef.h" 1 3 4
# 34 "/usr/include/bits/mathdef.h" 3 4
typedef float float_t;
typedef double double_t;
# 47 "/usr/include/math.h" 2 3 4
# 70 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 52 "/usr/include/bits/mathcalls.h" 3 4


extern double acos (double __x) throw (); extern double __acos (double __x) throw ();

extern double asin (double __x) throw (); extern double __asin (double __x) throw ();

extern double atan (double __x) throw (); extern double __atan (double __x) throw ();

extern double atan2 (double __y, double __x) throw (); extern double __atan2 (double __y, double __x) throw ();


extern double cos (double __x) throw (); extern double __cos (double __x) throw ();

extern double sin (double __x) throw (); extern double __sin (double __x) throw ();

extern double tan (double __x) throw (); extern double __tan (double __x) throw ();




extern double cosh (double __x) throw (); extern double __cosh (double __x) throw ();

extern double sinh (double __x) throw (); extern double __sinh (double __x) throw ();

extern double tanh (double __x) throw (); extern double __tanh (double __x) throw ();




extern void sincos (double __x, double *__sinx, double *__cosx) throw (); extern void __sincos (double __x, double *__sinx, double *__cosx) throw ()
                                                           ;





extern double acosh (double __x) throw (); extern double __acosh (double __x) throw ();

extern double asinh (double __x) throw (); extern double __asinh (double __x) throw ();

extern double atanh (double __x) throw (); extern double __atanh (double __x) throw ();







extern double exp (double __x) throw (); extern double __exp (double __x) throw ();


extern double frexp (double __x, int *__exponent) throw (); extern double __frexp (double __x, int *__exponent) throw ();


extern double ldexp (double __x, int __exponent) throw (); extern double __ldexp (double __x, int __exponent) throw ();


extern double log (double __x) throw (); extern double __log (double __x) throw ();


extern double log10 (double __x) throw (); extern double __log10 (double __x) throw ();


extern double modf (double __x, double *__iptr) throw (); extern double __modf (double __x, double *__iptr) throw ()
     __attribute__ ((__nonnull__ (2)));




extern double exp10 (double __x) throw (); extern double __exp10 (double __x) throw ();

extern double pow10 (double __x) throw (); extern double __pow10 (double __x) throw ();





extern double expm1 (double __x) throw (); extern double __expm1 (double __x) throw ();


extern double log1p (double __x) throw (); extern double __log1p (double __x) throw ();


extern double logb (double __x) throw (); extern double __logb (double __x) throw ();






extern double exp2 (double __x) throw (); extern double __exp2 (double __x) throw ();


extern double log2 (double __x) throw (); extern double __log2 (double __x) throw ();








extern double pow (double __x, double __y) throw (); extern double __pow (double __x, double __y) throw ();


extern double sqrt (double __x) throw (); extern double __sqrt (double __x) throw ();





extern double hypot (double __x, double __y) throw (); extern double __hypot (double __x, double __y) throw ();






extern double cbrt (double __x) throw (); extern double __cbrt (double __x) throw ();








extern double ceil (double __x) throw () __attribute__ ((__const__)); extern double __ceil (double __x) throw () __attribute__ ((__const__));


extern double fabs (double __x) throw () __attribute__ ((__const__)); extern double __fabs (double __x) throw () __attribute__ ((__const__));


extern double floor (double __x) throw () __attribute__ ((__const__)); extern double __floor (double __x) throw () __attribute__ ((__const__));


extern double fmod (double __x, double __y) throw (); extern double __fmod (double __x, double __y) throw ();




extern int __isinf (double __value) throw () __attribute__ ((__const__));


extern int __finite (double __value) throw () __attribute__ ((__const__));





extern int isinf (double __value) throw () __attribute__ ((__const__));


extern int finite (double __value) throw () __attribute__ ((__const__));


extern double drem (double __x, double __y) throw (); extern double __drem (double __x, double __y) throw ();



extern double significand (double __x) throw (); extern double __significand (double __x) throw ();





extern double copysign (double __x, double __y) throw () __attribute__ ((__const__)); extern double __copysign (double __x, double __y) throw () __attribute__ ((__const__));






extern double nan (const char *__tagb) throw () __attribute__ ((__const__)); extern double __nan (const char *__tagb) throw () __attribute__ ((__const__));





extern int __isnan (double __value) throw () __attribute__ ((__const__));



extern int isnan (double __value) throw () __attribute__ ((__const__));


extern double j0 (double) throw (); extern double __j0 (double) throw ();
extern double j1 (double) throw (); extern double __j1 (double) throw ();
extern double jn (int, double) throw (); extern double __jn (int, double) throw ();
extern double y0 (double) throw (); extern double __y0 (double) throw ();
extern double y1 (double) throw (); extern double __y1 (double) throw ();
extern double yn (int, double) throw (); extern double __yn (int, double) throw ();






extern double erf (double) throw (); extern double __erf (double) throw ();
extern double erfc (double) throw (); extern double __erfc (double) throw ();
extern double lgamma (double) throw (); extern double __lgamma (double) throw ();






extern double tgamma (double) throw (); extern double __tgamma (double) throw ();





extern double gamma (double) throw (); extern double __gamma (double) throw ();






extern double lgamma_r (double, int *__signgamp) throw (); extern double __lgamma_r (double, int *__signgamp) throw ();







extern double rint (double __x) throw (); extern double __rint (double __x) throw ();


extern double nextafter (double __x, double __y) throw () __attribute__ ((__const__)); extern double __nextafter (double __x, double __y) throw () __attribute__ ((__const__));

extern double nexttoward (double __x, long double __y) throw () __attribute__ ((__const__)); extern double __nexttoward (double __x, long double __y) throw () __attribute__ ((__const__));



extern double remainder (double __x, double __y) throw (); extern double __remainder (double __x, double __y) throw ();



extern double scalbn (double __x, int __n) throw (); extern double __scalbn (double __x, int __n) throw ();



extern int ilogb (double __x) throw (); extern int __ilogb (double __x) throw ();




extern double scalbln (double __x, long int __n) throw (); extern double __scalbln (double __x, long int __n) throw ();



extern double nearbyint (double __x) throw (); extern double __nearbyint (double __x) throw ();



extern double round (double __x) throw () __attribute__ ((__const__)); extern double __round (double __x) throw () __attribute__ ((__const__));



extern double trunc (double __x) throw () __attribute__ ((__const__)); extern double __trunc (double __x) throw () __attribute__ ((__const__));




extern double remquo (double __x, double __y, int *__quo) throw (); extern double __remquo (double __x, double __y, int *__quo) throw ();






extern long int lrint (double __x) throw (); extern long int __lrint (double __x) throw ();
extern long long int llrint (double __x) throw (); extern long long int __llrint (double __x) throw ();



extern long int lround (double __x) throw (); extern long int __lround (double __x) throw ();
extern long long int llround (double __x) throw (); extern long long int __llround (double __x) throw ();



extern double fdim (double __x, double __y) throw (); extern double __fdim (double __x, double __y) throw ();


extern double fmax (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fmax (double __x, double __y) throw () __attribute__ ((__const__));


extern double fmin (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fmin (double __x, double __y) throw () __attribute__ ((__const__));



extern int __fpclassify (double __value) throw ()
     __attribute__ ((__const__));


extern int __signbit (double __value) throw ()
     __attribute__ ((__const__));



extern double fma (double __x, double __y, double __z) throw (); extern double __fma (double __x, double __y, double __z) throw ();








extern double scalb (double __x, double __n) throw (); extern double __scalb (double __x, double __n) throw ();
# 71 "/usr/include/math.h" 2 3 4
# 89 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 52 "/usr/include/bits/mathcalls.h" 3 4


extern float acosf (float __x) throw (); extern float __acosf (float __x) throw ();

extern float asinf (float __x) throw (); extern float __asinf (float __x) throw ();

extern float atanf (float __x) throw (); extern float __atanf (float __x) throw ();

extern float atan2f (float __y, float __x) throw (); extern float __atan2f (float __y, float __x) throw ();


extern float cosf (float __x) throw (); extern float __cosf (float __x) throw ();

extern float sinf (float __x) throw (); extern float __sinf (float __x) throw ();

extern float tanf (float __x) throw (); extern float __tanf (float __x) throw ();




extern float coshf (float __x) throw (); extern float __coshf (float __x) throw ();

extern float sinhf (float __x) throw (); extern float __sinhf (float __x) throw ();

extern float tanhf (float __x) throw (); extern float __tanhf (float __x) throw ();




extern void sincosf (float __x, float *__sinx, float *__cosx) throw (); extern void __sincosf (float __x, float *__sinx, float *__cosx) throw ()
                                                           ;





extern float acoshf (float __x) throw (); extern float __acoshf (float __x) throw ();

extern float asinhf (float __x) throw (); extern float __asinhf (float __x) throw ();

extern float atanhf (float __x) throw (); extern float __atanhf (float __x) throw ();







extern float expf (float __x) throw (); extern float __expf (float __x) throw ();


extern float frexpf (float __x, int *__exponent) throw (); extern float __frexpf (float __x, int *__exponent) throw ();


extern float ldexpf (float __x, int __exponent) throw (); extern float __ldexpf (float __x, int __exponent) throw ();


extern float logf (float __x) throw (); extern float __logf (float __x) throw ();


extern float log10f (float __x) throw (); extern float __log10f (float __x) throw ();


extern float modff (float __x, float *__iptr) throw (); extern float __modff (float __x, float *__iptr) throw ()
     __attribute__ ((__nonnull__ (2)));




extern float exp10f (float __x) throw (); extern float __exp10f (float __x) throw ();

extern float pow10f (float __x) throw (); extern float __pow10f (float __x) throw ();





extern float expm1f (float __x) throw (); extern float __expm1f (float __x) throw ();


extern float log1pf (float __x) throw (); extern float __log1pf (float __x) throw ();


extern float logbf (float __x) throw (); extern float __logbf (float __x) throw ();






extern float exp2f (float __x) throw (); extern float __exp2f (float __x) throw ();


extern float log2f (float __x) throw (); extern float __log2f (float __x) throw ();








extern float powf (float __x, float __y) throw (); extern float __powf (float __x, float __y) throw ();


extern float sqrtf (float __x) throw (); extern float __sqrtf (float __x) throw ();





extern float hypotf (float __x, float __y) throw (); extern float __hypotf (float __x, float __y) throw ();






extern float cbrtf (float __x) throw (); extern float __cbrtf (float __x) throw ();








extern float ceilf (float __x) throw () __attribute__ ((__const__)); extern float __ceilf (float __x) throw () __attribute__ ((__const__));


extern float fabsf (float __x) throw () __attribute__ ((__const__)); extern float __fabsf (float __x) throw () __attribute__ ((__const__));


extern float floorf (float __x) throw () __attribute__ ((__const__)); extern float __floorf (float __x) throw () __attribute__ ((__const__));


extern float fmodf (float __x, float __y) throw (); extern float __fmodf (float __x, float __y) throw ();




extern int __isinff (float __value) throw () __attribute__ ((__const__));


extern int __finitef (float __value) throw () __attribute__ ((__const__));





extern int isinff (float __value) throw () __attribute__ ((__const__));


extern int finitef (float __value) throw () __attribute__ ((__const__));


extern float dremf (float __x, float __y) throw (); extern float __dremf (float __x, float __y) throw ();



extern float significandf (float __x) throw (); extern float __significandf (float __x) throw ();





extern float copysignf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __copysignf (float __x, float __y) throw () __attribute__ ((__const__));






extern float nanf (const char *__tagb) throw () __attribute__ ((__const__)); extern float __nanf (const char *__tagb) throw () __attribute__ ((__const__));





extern int __isnanf (float __value) throw () __attribute__ ((__const__));



extern int isnanf (float __value) throw () __attribute__ ((__const__));


extern float j0f (float) throw (); extern float __j0f (float) throw ();
extern float j1f (float) throw (); extern float __j1f (float) throw ();
extern float jnf (int, float) throw (); extern float __jnf (int, float) throw ();
extern float y0f (float) throw (); extern float __y0f (float) throw ();
extern float y1f (float) throw (); extern float __y1f (float) throw ();
extern float ynf (int, float) throw (); extern float __ynf (int, float) throw ();






extern float erff (float) throw (); extern float __erff (float) throw ();
extern float erfcf (float) throw (); extern float __erfcf (float) throw ();
extern float lgammaf (float) throw (); extern float __lgammaf (float) throw ();






extern float tgammaf (float) throw (); extern float __tgammaf (float) throw ();





extern float gammaf (float) throw (); extern float __gammaf (float) throw ();






extern float lgammaf_r (float, int *__signgamp) throw (); extern float __lgammaf_r (float, int *__signgamp) throw ();







extern float rintf (float __x) throw (); extern float __rintf (float __x) throw ();


extern float nextafterf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __nextafterf (float __x, float __y) throw () __attribute__ ((__const__));

extern float nexttowardf (float __x, long double __y) throw () __attribute__ ((__const__)); extern float __nexttowardf (float __x, long double __y) throw () __attribute__ ((__const__));



extern float remainderf (float __x, float __y) throw (); extern float __remainderf (float __x, float __y) throw ();



extern float scalbnf (float __x, int __n) throw (); extern float __scalbnf (float __x, int __n) throw ();



extern int ilogbf (float __x) throw (); extern int __ilogbf (float __x) throw ();




extern float scalblnf (float __x, long int __n) throw (); extern float __scalblnf (float __x, long int __n) throw ();



extern float nearbyintf (float __x) throw (); extern float __nearbyintf (float __x) throw ();



extern float roundf (float __x) throw () __attribute__ ((__const__)); extern float __roundf (float __x) throw () __attribute__ ((__const__));



extern float truncf (float __x) throw () __attribute__ ((__const__)); extern float __truncf (float __x) throw () __attribute__ ((__const__));




extern float remquof (float __x, float __y, int *__quo) throw (); extern float __remquof (float __x, float __y, int *__quo) throw ();






extern long int lrintf (float __x) throw (); extern long int __lrintf (float __x) throw ();
extern long long int llrintf (float __x) throw (); extern long long int __llrintf (float __x) throw ();



extern long int lroundf (float __x) throw (); extern long int __lroundf (float __x) throw ();
extern long long int llroundf (float __x) throw (); extern long long int __llroundf (float __x) throw ();



extern float fdimf (float __x, float __y) throw (); extern float __fdimf (float __x, float __y) throw ();


extern float fmaxf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fmaxf (float __x, float __y) throw () __attribute__ ((__const__));


extern float fminf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fminf (float __x, float __y) throw () __attribute__ ((__const__));



extern int __fpclassifyf (float __value) throw ()
     __attribute__ ((__const__));


extern int __signbitf (float __value) throw ()
     __attribute__ ((__const__));



extern float fmaf (float __x, float __y, float __z) throw (); extern float __fmaf (float __x, float __y, float __z) throw ();








extern float scalbf (float __x, float __n) throw (); extern float __scalbf (float __x, float __n) throw ();
# 90 "/usr/include/math.h" 2 3 4
# 133 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 52 "/usr/include/bits/mathcalls.h" 3 4


extern long double acosl (long double __x) throw (); extern long double __acosl (long double __x) throw ();

extern long double asinl (long double __x) throw (); extern long double __asinl (long double __x) throw ();

extern long double atanl (long double __x) throw (); extern long double __atanl (long double __x) throw ();

extern long double atan2l (long double __y, long double __x) throw (); extern long double __atan2l (long double __y, long double __x) throw ();


extern long double cosl (long double __x) throw (); extern long double __cosl (long double __x) throw ();

extern long double sinl (long double __x) throw (); extern long double __sinl (long double __x) throw ();

extern long double tanl (long double __x) throw (); extern long double __tanl (long double __x) throw ();




extern long double coshl (long double __x) throw (); extern long double __coshl (long double __x) throw ();

extern long double sinhl (long double __x) throw (); extern long double __sinhl (long double __x) throw ();

extern long double tanhl (long double __x) throw (); extern long double __tanhl (long double __x) throw ();




extern void sincosl (long double __x, long double *__sinx, long double *__cosx) throw (); extern void __sincosl (long double __x, long double *__sinx, long double *__cosx) throw ()
                                                           ;





extern long double acoshl (long double __x) throw (); extern long double __acoshl (long double __x) throw ();

extern long double asinhl (long double __x) throw (); extern long double __asinhl (long double __x) throw ();

extern long double atanhl (long double __x) throw (); extern long double __atanhl (long double __x) throw ();







extern long double expl (long double __x) throw (); extern long double __expl (long double __x) throw ();


extern long double frexpl (long double __x, int *__exponent) throw (); extern long double __frexpl (long double __x, int *__exponent) throw ();


extern long double ldexpl (long double __x, int __exponent) throw (); extern long double __ldexpl (long double __x, int __exponent) throw ();


extern long double logl (long double __x) throw (); extern long double __logl (long double __x) throw ();


extern long double log10l (long double __x) throw (); extern long double __log10l (long double __x) throw ();


extern long double modfl (long double __x, long double *__iptr) throw (); extern long double __modfl (long double __x, long double *__iptr) throw ()
     __attribute__ ((__nonnull__ (2)));




extern long double exp10l (long double __x) throw (); extern long double __exp10l (long double __x) throw ();

extern long double pow10l (long double __x) throw (); extern long double __pow10l (long double __x) throw ();





extern long double expm1l (long double __x) throw (); extern long double __expm1l (long double __x) throw ();


extern long double log1pl (long double __x) throw (); extern long double __log1pl (long double __x) throw ();


extern long double logbl (long double __x) throw (); extern long double __logbl (long double __x) throw ();






extern long double exp2l (long double __x) throw (); extern long double __exp2l (long double __x) throw ();


extern long double log2l (long double __x) throw (); extern long double __log2l (long double __x) throw ();








extern long double powl (long double __x, long double __y) throw (); extern long double __powl (long double __x, long double __y) throw ();


extern long double sqrtl (long double __x) throw (); extern long double __sqrtl (long double __x) throw ();





extern long double hypotl (long double __x, long double __y) throw (); extern long double __hypotl (long double __x, long double __y) throw ();






extern long double cbrtl (long double __x) throw (); extern long double __cbrtl (long double __x) throw ();








extern long double ceill (long double __x) throw () __attribute__ ((__const__)); extern long double __ceill (long double __x) throw () __attribute__ ((__const__));


extern long double fabsl (long double __x) throw () __attribute__ ((__const__)); extern long double __fabsl (long double __x) throw () __attribute__ ((__const__));


extern long double floorl (long double __x) throw () __attribute__ ((__const__)); extern long double __floorl (long double __x) throw () __attribute__ ((__const__));


extern long double fmodl (long double __x, long double __y) throw (); extern long double __fmodl (long double __x, long double __y) throw ();




extern int __isinfl (long double __value) throw () __attribute__ ((__const__));


extern int __finitel (long double __value) throw () __attribute__ ((__const__));





extern int isinfl (long double __value) throw () __attribute__ ((__const__));


extern int finitel (long double __value) throw () __attribute__ ((__const__));


extern long double dreml (long double __x, long double __y) throw (); extern long double __dreml (long double __x, long double __y) throw ();



extern long double significandl (long double __x) throw (); extern long double __significandl (long double __x) throw ();





extern long double copysignl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __copysignl (long double __x, long double __y) throw () __attribute__ ((__const__));






extern long double nanl (const char *__tagb) throw () __attribute__ ((__const__)); extern long double __nanl (const char *__tagb) throw () __attribute__ ((__const__));





extern int __isnanl (long double __value) throw () __attribute__ ((__const__));



extern int isnanl (long double __value) throw () __attribute__ ((__const__));


extern long double j0l (long double) throw (); extern long double __j0l (long double) throw ();
extern long double j1l (long double) throw (); extern long double __j1l (long double) throw ();
extern long double jnl (int, long double) throw (); extern long double __jnl (int, long double) throw ();
extern long double y0l (long double) throw (); extern long double __y0l (long double) throw ();
extern long double y1l (long double) throw (); extern long double __y1l (long double) throw ();
extern long double ynl (int, long double) throw (); extern long double __ynl (int, long double) throw ();






extern long double erfl (long double) throw (); extern long double __erfl (long double) throw ();
extern long double erfcl (long double) throw (); extern long double __erfcl (long double) throw ();
extern long double lgammal (long double) throw (); extern long double __lgammal (long double) throw ();






extern long double tgammal (long double) throw (); extern long double __tgammal (long double) throw ();





extern long double gammal (long double) throw (); extern long double __gammal (long double) throw ();






extern long double lgammal_r (long double, int *__signgamp) throw (); extern long double __lgammal_r (long double, int *__signgamp) throw ();







extern long double rintl (long double __x) throw (); extern long double __rintl (long double __x) throw ();


extern long double nextafterl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __nextafterl (long double __x, long double __y) throw () __attribute__ ((__const__));

extern long double nexttowardl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __nexttowardl (long double __x, long double __y) throw () __attribute__ ((__const__));



extern long double remainderl (long double __x, long double __y) throw (); extern long double __remainderl (long double __x, long double __y) throw ();



extern long double scalbnl (long double __x, int __n) throw (); extern long double __scalbnl (long double __x, int __n) throw ();



extern int ilogbl (long double __x) throw (); extern int __ilogbl (long double __x) throw ();




extern long double scalblnl (long double __x, long int __n) throw (); extern long double __scalblnl (long double __x, long int __n) throw ();



extern long double nearbyintl (long double __x) throw (); extern long double __nearbyintl (long double __x) throw ();



extern long double roundl (long double __x) throw () __attribute__ ((__const__)); extern long double __roundl (long double __x) throw () __attribute__ ((__const__));



extern long double truncl (long double __x) throw () __attribute__ ((__const__)); extern long double __truncl (long double __x) throw () __attribute__ ((__const__));




extern long double remquol (long double __x, long double __y, int *__quo) throw (); extern long double __remquol (long double __x, long double __y, int *__quo) throw ();






extern long int lrintl (long double __x) throw (); extern long int __lrintl (long double __x) throw ();
extern long long int llrintl (long double __x) throw (); extern long long int __llrintl (long double __x) throw ();



extern long int lroundl (long double __x) throw (); extern long int __lroundl (long double __x) throw ();
extern long long int llroundl (long double __x) throw (); extern long long int __llroundl (long double __x) throw ();



extern long double fdiml (long double __x, long double __y) throw (); extern long double __fdiml (long double __x, long double __y) throw ();


extern long double fmaxl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fmaxl (long double __x, long double __y) throw () __attribute__ ((__const__));


extern long double fminl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fminl (long double __x, long double __y) throw () __attribute__ ((__const__));



extern int __fpclassifyl (long double __value) throw ()
     __attribute__ ((__const__));


extern int __signbitl (long double __value) throw ()
     __attribute__ ((__const__));



extern long double fmal (long double __x, long double __y, long double __z) throw (); extern long double __fmal (long double __x, long double __y, long double __z) throw ();








extern long double scalbl (long double __x, long double __n) throw (); extern long double __scalbl (long double __x, long double __n) throw ();
# 134 "/usr/include/math.h" 2 3 4
# 149 "/usr/include/math.h" 3 4
extern int signgam;
# 190 "/usr/include/math.h" 3 4
enum
  {
    FP_NAN =

      0,
    FP_INFINITE =

      1,
    FP_ZERO =

      2,
    FP_SUBNORMAL =

      3,
    FP_NORMAL =

      4
  };
# 288 "/usr/include/math.h" 3 4
typedef enum
{
  _IEEE_ = -1,
  _SVID_,
  _XOPEN_,
  _POSIX_,
  _ISOC_
} _LIB_VERSION_TYPE;




extern _LIB_VERSION_TYPE _LIB_VERSION;
# 311 "/usr/include/math.h" 3 4
struct __exception



  {
    int type;
    char *name;
    double arg1;
    double arg2;
    double retval;
  };


extern int matherr (struct __exception *__exc) throw ();
# 475 "/usr/include/math.h" 3 4
}
# 8850 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 2



# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 1 3
# 39 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 3
       
# 40 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 3


# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 1 3
# 35 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 3
       
# 36 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 3
# 68 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{


  template<typename _Iterator, typename _Container>
    class __normal_iterator;


}

namespace std __attribute__ ((__visibility__ ("default")))
{


  struct __true_type { };
  struct __false_type { };

  template<bool>
    struct __truth_type
    { typedef __false_type __type; };

  template<>
    struct __truth_type<true>
    { typedef __true_type __type; };



  template<class _Sp, class _Tp>
    struct __traitor
    {
      enum { __value = bool(_Sp::__value) || bool(_Tp::__value) };
      typedef typename __truth_type<__value>::__type __type;
    };


  template<typename, typename>
    struct __are_same
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Tp>
    struct __are_same<_Tp, _Tp>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<typename _Tp>
    struct __is_void
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_void<void>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_integer
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };




  template<>
    struct __is_integer<bool>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<signed char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<>
    struct __is_integer<wchar_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };
# 198 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 3
  template<>
    struct __is_integer<short>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned short>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<int>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned int>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<long long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned long long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_floating
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };


  template<>
    struct __is_floating<float>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_floating<double>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_floating<long double>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_pointer
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Tp>
    struct __is_pointer<_Tp*>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_normal_iterator
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Iterator, typename _Container>
    struct __is_normal_iterator< __gnu_cxx::__normal_iterator<_Iterator,
             _Container> >
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_arithmetic
    : public __traitor<__is_integer<_Tp>, __is_floating<_Tp> >
    { };




  template<typename _Tp>
    struct __is_scalar
    : public __traitor<__is_arithmetic<_Tp>, __is_pointer<_Tp> >
    { };




  template<typename _Tp>
    struct __is_char
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_char<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<>
    struct __is_char<wchar_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<typename _Tp>
    struct __is_byte
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_byte<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_byte<signed char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_byte<unsigned char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_move_iterator
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };
# 413 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/bits/cpp_type_traits.h" 3

}
# 43 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 2 3
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 1 3
# 32 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
       
# 33 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3




namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{



  template<bool, typename>
    struct __enable_if
    { };

  template<typename _Tp>
    struct __enable_if<true, _Tp>
    { typedef _Tp __type; };



  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    struct __conditional_type
    { typedef _Iftrue __type; };

  template<typename _Iftrue, typename _Iffalse>
    struct __conditional_type<false, _Iftrue, _Iffalse>
    { typedef _Iffalse __type; };



  template<typename _Tp>
    struct __add_unsigned
    {
    private:
      typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;

    public:
      typedef typename __if_type::__type __type;
    };

  template<>
    struct __add_unsigned<char>
    { typedef unsigned char __type; };

  template<>
    struct __add_unsigned<signed char>
    { typedef unsigned char __type; };

  template<>
    struct __add_unsigned<short>
    { typedef unsigned short __type; };

  template<>
    struct __add_unsigned<int>
    { typedef unsigned int __type; };

  template<>
    struct __add_unsigned<long>
    { typedef unsigned long __type; };

  template<>
    struct __add_unsigned<long long>
    { typedef unsigned long long __type; };


  template<>
    struct __add_unsigned<bool>;

  template<>
    struct __add_unsigned<wchar_t>;



  template<typename _Tp>
    struct __remove_unsigned
    {
    private:
      typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;

    public:
      typedef typename __if_type::__type __type;
    };

  template<>
    struct __remove_unsigned<char>
    { typedef signed char __type; };

  template<>
    struct __remove_unsigned<unsigned char>
    { typedef signed char __type; };

  template<>
    struct __remove_unsigned<unsigned short>
    { typedef short __type; };

  template<>
    struct __remove_unsigned<unsigned int>
    { typedef int __type; };

  template<>
    struct __remove_unsigned<unsigned long>
    { typedef long __type; };

  template<>
    struct __remove_unsigned<unsigned long long>
    { typedef long long __type; };


  template<>
    struct __remove_unsigned<bool>;

  template<>
    struct __remove_unsigned<wchar_t>;



  template<typename _Type>
    inline
   
# 150 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
   
# 149 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
           bool
    __is_null_pointer(_Type* __ptr)
    { return __ptr == 0; }

  template<typename _Type>
    inline
   
# 155 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
   
# 154 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
           bool
    __is_null_pointer(_Type)
    { return false; }
# 165 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/ext/type_traits.h" 3
  template<typename _Tp, bool = std::__is_integer<_Tp>::__value>
    struct __promote
    { typedef double __type; };




  template<typename _Tp>
    struct __promote<_Tp, false>
    { };

  template<>
    struct __promote<long double>
    { typedef long double __type; };

  template<>
    struct __promote<double>
    { typedef double __type; };

  template<>
    struct __promote<float>
    { typedef float __type; };

  template<typename _Tp, typename _Up,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type>
    struct __promote_2
    {
      typedef __typeof__(_Tp2() + _Up2()) __type;
    };

  template<typename _Tp, typename _Up, typename _Vp,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type,
           typename _Vp2 = typename __promote<_Vp>::__type>
    struct __promote_3
    {
      typedef __typeof__(_Tp2() + _Up2() + _Vp2()) __type;
    };

  template<typename _Tp, typename _Up, typename _Vp, typename _Wp,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type,
           typename _Vp2 = typename __promote<_Vp>::__type,
           typename _Wp2 = typename __promote<_Wp>::__type>
    struct __promote_4
    {
      typedef __typeof__(_Tp2() + _Up2() + _Vp2() + _Wp2()) __type;
    };


}
# 44 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 2 3
# 75 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 3
namespace std __attribute__ ((__visibility__ ("default")))
{



  inline double
  abs(double __x)
  { return __builtin_fabs(__x); }



  inline float
  abs(float __x)
  { return __builtin_fabsf(__x); }

  inline long double
  abs(long double __x)
  { return __builtin_fabsl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    abs(_Tp __x)
    { return __builtin_fabs(__x); }

  using ::acos;


  inline float
  acos(float __x)
  { return __builtin_acosf(__x); }

  inline long double
  acos(long double __x)
  { return __builtin_acosl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    acos(_Tp __x)
    { return __builtin_acos(__x); }

  using ::asin;


  inline float
  asin(float __x)
  { return __builtin_asinf(__x); }

  inline long double
  asin(long double __x)
  { return __builtin_asinl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    asin(_Tp __x)
    { return __builtin_asin(__x); }

  using ::atan;


  inline float
  atan(float __x)
  { return __builtin_atanf(__x); }

  inline long double
  atan(long double __x)
  { return __builtin_atanl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    atan(_Tp __x)
    { return __builtin_atan(__x); }

  using ::atan2;


  inline float
  atan2(float __y, float __x)
  { return __builtin_atan2f(__y, __x); }

  inline long double
  atan2(long double __y, long double __x)
  { return __builtin_atan2l(__y, __x); }


  template<typename _Tp, typename _Up>
    inline
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    atan2(_Tp __y, _Up __x)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return atan2(__type(__y), __type(__x));
    }

  using ::ceil;


  inline float
  ceil(float __x)
  { return __builtin_ceilf(__x); }

  inline long double
  ceil(long double __x)
  { return __builtin_ceill(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    ceil(_Tp __x)
    { return __builtin_ceil(__x); }

  using ::cos;


  inline float
  cos(float __x)
  { return __builtin_cosf(__x); }

  inline long double
  cos(long double __x)
  { return __builtin_cosl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    cos(_Tp __x)
    { return __builtin_cos(__x); }

  using ::cosh;


  inline float
  cosh(float __x)
  { return __builtin_coshf(__x); }

  inline long double
  cosh(long double __x)
  { return __builtin_coshl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    cosh(_Tp __x)
    { return __builtin_cosh(__x); }

  using ::exp;


  inline float
  exp(float __x)
  { return __builtin_expf(__x); }

  inline long double
  exp(long double __x)
  { return __builtin_expl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    exp(_Tp __x)
    { return __builtin_exp(__x); }

  using ::fabs;


  inline float
  fabs(float __x)
  { return __builtin_fabsf(__x); }

  inline long double
  fabs(long double __x)
  { return __builtin_fabsl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    fabs(_Tp __x)
    { return __builtin_fabs(__x); }

  using ::floor;


  inline float
  floor(float __x)
  { return __builtin_floorf(__x); }

  inline long double
  floor(long double __x)
  { return __builtin_floorl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    floor(_Tp __x)
    { return __builtin_floor(__x); }

  using ::fmod;


  inline float
  fmod(float __x, float __y)
  { return __builtin_fmodf(__x, __y); }

  inline long double
  fmod(long double __x, long double __y)
  { return __builtin_fmodl(__x, __y); }


  template<typename _Tp, typename _Up>
    inline
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmod(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return fmod(__type(__x), __type(__y));
    }

  using ::frexp;


  inline float
  frexp(float __x, int* __exp)
  { return __builtin_frexpf(__x, __exp); }

  inline long double
  frexp(long double __x, int* __exp)
  { return __builtin_frexpl(__x, __exp); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    frexp(_Tp __x, int* __exp)
    { return __builtin_frexp(__x, __exp); }

  using ::ldexp;


  inline float
  ldexp(float __x, int __exp)
  { return __builtin_ldexpf(__x, __exp); }

  inline long double
  ldexp(long double __x, int __exp)
  { return __builtin_ldexpl(__x, __exp); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    ldexp(_Tp __x, int __exp)
    { return __builtin_ldexp(__x, __exp); }

  using ::log;


  inline float
  log(float __x)
  { return __builtin_logf(__x); }

  inline long double
  log(long double __x)
  { return __builtin_logl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    log(_Tp __x)
    { return __builtin_log(__x); }

  using ::log10;


  inline float
  log10(float __x)
  { return __builtin_log10f(__x); }

  inline long double
  log10(long double __x)
  { return __builtin_log10l(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    log10(_Tp __x)
    { return __builtin_log10(__x); }

  using ::modf;


  inline float
  modf(float __x, float* __iptr)
  { return __builtin_modff(__x, __iptr); }

  inline long double
  modf(long double __x, long double* __iptr)
  { return __builtin_modfl(__x, __iptr); }


  using ::pow;


  inline float
  pow(float __x, float __y)
  { return __builtin_powf(__x, __y); }

  inline long double
  pow(long double __x, long double __y)
  { return __builtin_powl(__x, __y); }




  inline double
  pow(double __x, int __i)
  { return __builtin_powi(__x, __i); }

  inline float
  pow(float __x, int __n)
  { return __builtin_powif(__x, __n); }

  inline long double
  pow(long double __x, int __n)
  { return __builtin_powil(__x, __n); }



  template<typename _Tp, typename _Up>
    inline
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    pow(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return pow(__type(__x), __type(__y));
    }

  using ::sin;


  inline float
  sin(float __x)
  { return __builtin_sinf(__x); }

  inline long double
  sin(long double __x)
  { return __builtin_sinl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sin(_Tp __x)
    { return __builtin_sin(__x); }

  using ::sinh;


  inline float
  sinh(float __x)
  { return __builtin_sinhf(__x); }

  inline long double
  sinh(long double __x)
  { return __builtin_sinhl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sinh(_Tp __x)
    { return __builtin_sinh(__x); }

  using ::sqrt;


  inline float
  sqrt(float __x)
  { return __builtin_sqrtf(__x); }

  inline long double
  sqrt(long double __x)
  { return __builtin_sqrtl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sqrt(_Tp __x)
    { return __builtin_sqrt(__x); }

  using ::tan;


  inline float
  tan(float __x)
  { return __builtin_tanf(__x); }

  inline long double
  tan(long double __x)
  { return __builtin_tanl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    tan(_Tp __x)
    { return __builtin_tan(__x); }

  using ::tanh;


  inline float
  tanh(float __x)
  { return __builtin_tanhf(__x); }

  inline long double
  tanh(long double __x)
  { return __builtin_tanhl(__x); }


  template<typename _Tp>
    inline
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    tanh(_Tp __x)
    { return __builtin_tanh(__x); }


}
# 555 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 3
namespace std __attribute__ ((__visibility__ ("default")))
{

# 806 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cmath" 3
  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    fpclassify(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_fpclassify(0, 1, 4,
      3, 2, __type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isfinite(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isfinite(__type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isinf(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isinf(__type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isnan(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isnan(__type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isnormal(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isnormal(__type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    signbit(_Tp __f)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_signbit(__type(__f));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isgreater(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isgreater(__type(__f1), __type(__f2));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isgreaterequal(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isgreaterequal(__type(__f1), __type(__f2));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isless(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isless(__type(__f1), __type(__f2));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    islessequal(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_islessequal(__type(__f1), __type(__f2));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    islessgreater(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_islessgreater(__type(__f1), __type(__f2));
    }

  template<typename _Tp>
    inline typename __gnu_cxx::__enable_if<__is_arithmetic<_Tp>::__value,
        int>::__type
    isunordered(_Tp __f1, _Tp __f2)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __builtin_isunordered(__type(__f1), __type(__f2));
    }




}
# 8854 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 2
# 1 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 1 3
# 39 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 3
       
# 40 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 3
# 114 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 3
namespace std __attribute__ ((__visibility__ ("default")))
{


  using ::div_t;
  using ::ldiv_t;

  using ::abort;
  using ::abs;
  using ::atexit;





  using ::atof;
  using ::atoi;
  using ::atol;
  using ::bsearch;
  using ::calloc;
  using ::div;
  using ::exit;
  using ::free;
  using ::getenv;
  using ::labs;
  using ::ldiv;
  using ::malloc;

  using ::mblen;
  using ::mbstowcs;
  using ::mbtowc;

  using ::qsort;





  using ::rand;
  using ::realloc;
  using ::srand;
  using ::strtod;
  using ::strtol;
  using ::strtoul;
  using ::system;

  using ::wcstombs;
  using ::wctomb;



  inline long
  abs(long __i) { return __builtin_labs(__i); }

  inline ldiv_t
  div(long __i, long __j) { return ldiv(__i, __j); }



  inline long long
  abs(long long __x) { return __builtin_llabs (__x); }



  inline __int128
  abs(__int128 __x) { return __x >= 0 ? __x : -__x; }



}
# 196 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 3
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{



  using ::lldiv_t;





  using ::_Exit;



  using ::llabs;

  inline lldiv_t
  div(long long __n, long long __d)
  { lldiv_t __q; __q.quot = __n / __d; __q.rem = __n % __d; return __q; }

  using ::lldiv;
# 228 "/usr/tce/packages/gcc/gcc-4.9.3/gnu/include/c++/4.9.3/cstdlib" 3
  using ::atoll;
  using ::strtoll;
  using ::strtoull;

  using ::strtof;
  using ::strtold;


}

namespace std
{

  using ::__gnu_cxx::lldiv_t;

  using ::__gnu_cxx::_Exit;

  using ::__gnu_cxx::llabs;
  using ::__gnu_cxx::div;
  using ::__gnu_cxx::lldiv;

  using ::__gnu_cxx::atoll;
  using ::__gnu_cxx::strtof;
  using ::__gnu_cxx::strtoll;
  using ::__gnu_cxx::strtoull;
  using ::__gnu_cxx::strtold;
}
# 8855 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 2
# 8960 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(float x);



__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(double x);

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(long double x);

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(float x);



__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(double x);

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(long double x);

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(float x);






__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(double x) throw();

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(long double x);

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(float x);






__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(double x) throw();

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(long double x);
# 9053 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
namespace std
{
  template<typename T> extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) T __pow_helper(T, int);
  template<typename T> extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) T __cmath_power(T, unsigned int);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;

using std::pow;

using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;
# 9448 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
namespace std {
# 9457 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int abs(long long int);
# 9467 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int abs(long int);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float abs(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double abs(double);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fabs(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float ceil(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float floor(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sqrt(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float pow(float, float);
# 9483 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float pow(float, int);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double pow(double, int);




extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log10(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fmod(float, float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float modf(float, float*);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float exp(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float frexp(float, int*);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float ldexp(float, int);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float asin(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sin(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sinh(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float acos(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cos(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cosh(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float atan(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float atan2(float, float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float tan(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float tanh(float);
# 9579 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
}
# 9722 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float logb(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int ilogb(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float scalbn(float a, int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float scalbln(float a, long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float exp2(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float expm1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log2(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log1p(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float acosh(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float asinh(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float atanh(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float hypot(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float norm3d(float a, float b, float c);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float norm4d(float a, float b, float c, float d);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cbrt(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erf(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfc(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float lgamma(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float tgamma(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float copysign(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float nextafter(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float remainder(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float remquo(float a, float b, int *quo);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float round(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int lround(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int llround(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float trunc(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float rint(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int lrint(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int llrint(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float nearbyint(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fdim(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fma(float a, float b, float c);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fmax(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fmin(float a, float b);
# 9831 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float exp10(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float rsqrt(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float rcbrt(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sinpi(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cospi(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void sincospi(float a, float *sptr, float *cptr);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void sincos(float a, float *sptr, float *cptr);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float j0(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float j1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float jn(int n, float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float y0(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float y1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float yn(int n, float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cyl_bessel_i0(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cyl_bessel_i1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfcinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float normcdfinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float normcdf(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfcx(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double copysign(double a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double copysign(float a, double b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(unsigned int a, unsigned int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(int a, unsigned int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(unsigned int a, int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int min(long int a, long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(unsigned long int a, unsigned long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(long int a, unsigned long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(unsigned long int a, long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int min(long long int a, long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(unsigned long long int a, unsigned long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(long long int a, unsigned long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(unsigned long long int a, long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float min(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(double a, double b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(float a, double b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(double a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(unsigned int a, unsigned int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(int a, unsigned int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(unsigned int a, int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int max(long int a, long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(unsigned long int a, unsigned long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(long int a, unsigned long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(unsigned long int a, long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int max(long long int a, long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(unsigned long long int a, unsigned long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(long long int a, unsigned long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(unsigned long long int a, long long int b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float max(float a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(double a, double b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(float a, double b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(double a, float b);
# 10222 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp" 2
# 316 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(float x) { return __signbitf(x); }



__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(double x) { return __signbit(x); }

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int signbit(long double x) { return __signbitl(x);}
# 333 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(float x) { return __finitef(x); }
# 348 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(double x) { return __finite(x); }
# 361 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isfinite(long double x) { return __finitel(x); }


__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(float x) { return __isnanf(x); }



__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(double x) throw() { return __isnan(x); }

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isnan(long double x) { return __isnanl(x); }

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(float x) { return __isinff(x); }



__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(double x) throw() { return __isinf(x); }

__inline__ __attribute__((always_inline)) __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) int isinf(long double x) { return __isinfl(x); }
# 584 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.hpp"
static inline __attribute__((host)) __attribute__((device)) float logb(float a)
{
  return logbf(a);
}

static inline __attribute__((host)) __attribute__((device)) int ilogb(float a)
{
  return ilogbf(a);
}

static inline __attribute__((host)) __attribute__((device)) float scalbn(float a, int b)
{
  return scalbnf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float scalbln(float a, long int b)
{
  return scalblnf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float exp2(float a)
{
  return exp2f(a);
}

static inline __attribute__((host)) __attribute__((device)) float expm1(float a)
{
  return expm1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float log2(float a)
{
  return log2f(a);
}

static inline __attribute__((host)) __attribute__((device)) float log1p(float a)
{
  return log1pf(a);
}

static inline __attribute__((host)) __attribute__((device)) float acosh(float a)
{
  return acoshf(a);
}

static inline __attribute__((host)) __attribute__((device)) float asinh(float a)
{
  return asinhf(a);
}

static inline __attribute__((host)) __attribute__((device)) float atanh(float a)
{
  return atanhf(a);
}

static inline __attribute__((host)) __attribute__((device)) float hypot(float a, float b)
{
  return hypotf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float norm3d(float a, float b, float c)
{
  return norm3df(a, b, c);
}

static inline __attribute__((host)) __attribute__((device)) float norm4d(float a, float b, float c, float d)
{
  return norm4df(a, b, c, d);
}

static inline __attribute__((host)) __attribute__((device)) float cbrt(float a)
{
  return cbrtf(a);
}

static inline __attribute__((host)) __attribute__((device)) float erf(float a)
{
  return erff(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfc(float a)
{
  return erfcf(a);
}

static inline __attribute__((host)) __attribute__((device)) float lgamma(float a)
{
  return lgammaf(a);
}

static inline __attribute__((host)) __attribute__((device)) float tgamma(float a)
{
  return tgammaf(a);
}

static inline __attribute__((host)) __attribute__((device)) float copysign(float a, float b)
{
  return copysignf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float nextafter(float a, float b)
{
  return nextafterf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float remainder(float a, float b)
{
  return remainderf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float remquo(float a, float b, int *quo)
{
  return remquof(a, b, quo);
}

static inline __attribute__((host)) __attribute__((device)) float round(float a)
{
  return roundf(a);
}

static inline __attribute__((host)) __attribute__((device)) long int lround(float a)
{
  return lroundf(a);
}

static inline __attribute__((host)) __attribute__((device)) long long int llround(float a)
{
  return llroundf(a);
}

static inline __attribute__((host)) __attribute__((device)) float trunc(float a)
{
  return truncf(a);
}

static inline __attribute__((host)) __attribute__((device)) float rint(float a)
{
  return rintf(a);
}

static inline __attribute__((host)) __attribute__((device)) long int lrint(float a)
{
  return lrintf(a);
}

static inline __attribute__((host)) __attribute__((device)) long long int llrint(float a)
{
  return llrintf(a);
}

static inline __attribute__((host)) __attribute__((device)) float nearbyint(float a)
{
  return nearbyintf(a);
}

static inline __attribute__((host)) __attribute__((device)) float fdim(float a, float b)
{
  return fdimf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float fma(float a, float b, float c)
{
  return fmaf(a, b, c);
}

static inline __attribute__((host)) __attribute__((device)) float fmax(float a, float b)
{
  return fmaxf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) float fmin(float a, float b)
{
  return fminf(a, b);
}







static inline __attribute__((host)) __attribute__((device)) float exp10(float a)
{
  return exp10f(a);
}

static inline __attribute__((host)) __attribute__((device)) float rsqrt(float a)
{
  return rsqrtf(a);
}

static inline __attribute__((host)) __attribute__((device)) float rcbrt(float a)
{
  return rcbrtf(a);
}

static inline __attribute__((host)) __attribute__((device)) float sinpi(float a)
{
  return sinpif(a);
}

static inline __attribute__((host)) __attribute__((device)) float cospi(float a)
{
  return cospif(a);
}

static inline __attribute__((host)) __attribute__((device)) void sincospi(float a, float *sptr, float *cptr)
{
  sincospif(a, sptr, cptr);
}

static inline __attribute__((host)) __attribute__((device)) void sincos(float a, float *sptr, float *cptr)
{
  sincosf(a, sptr, cptr);
}

static inline __attribute__((host)) __attribute__((device)) float j0(float a)
{
  return j0f(a);
}

static inline __attribute__((host)) __attribute__((device)) float j1(float a)
{
  return j1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float jn(int n, float a)
{
  return jnf(n, a);
}

static inline __attribute__((host)) __attribute__((device)) float y0(float a)
{
  return y0f(a);
}

static inline __attribute__((host)) __attribute__((device)) float y1(float a)
{
  return y1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float yn(int n, float a)
{
  return ynf(n, a);
}

static inline __attribute__((host)) __attribute__((device)) float cyl_bessel_i0(float a)
{
  return cyl_bessel_i0f(a);
}

static inline __attribute__((host)) __attribute__((device)) float cyl_bessel_i1(float a)
{
  return cyl_bessel_i1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfinv(float a)
{
  return erfinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfcinv(float a)
{
  return erfcinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float normcdfinv(float a)
{
  return normcdfinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float normcdf(float a)
{
  return normcdff(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfcx(float a)
{
  return erfcxf(a);
}

static inline __attribute__((host)) __attribute__((device)) double copysign(double a, float b)
{
  return copysign(a, (double)b);
}

static inline __attribute__((host)) __attribute__((device)) double copysign(float a, double b)
{
  return copysign((double)a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(unsigned int a, unsigned int b)
{
  return umin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(int a, unsigned int b)
{
  return umin((unsigned int)a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(unsigned int a, int b)
{
  return umin(a, (unsigned int)b);
}

static inline __attribute__((host)) __attribute__((device)) long int min(long int a, long int b)
{





  if (sizeof(long int) == sizeof(int)) {



    return (long int)min((int)a, (int)b);
  } else {
    return (long int)llmin((long long int)a, (long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(unsigned long int a, unsigned long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umin((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmin((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(long int a, unsigned long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umin((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmin((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(unsigned long int a, long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umin((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmin((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) long long int min(long long int a, long long int b)
{
  return llmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(unsigned long long int a, unsigned long long int b)
{
  return ullmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(long long int a, unsigned long long int b)
{
  return ullmin((unsigned long long int)a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(unsigned long long int a, long long int b)
{
  return ullmin(a, (unsigned long long int)b);
}

static inline __attribute__((host)) __attribute__((device)) float min(float a, float b)
{
  return fminf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double min(double a, double b)
{
  return fmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double min(float a, double b)
{
  return fmin((double)a, b);
}

static inline __attribute__((host)) __attribute__((device)) double min(double a, float b)
{
  return fmin(a, (double)b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(unsigned int a, unsigned int b)
{
  return umax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(int a, unsigned int b)
{
  return umax((unsigned int)a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(unsigned int a, int b)
{
  return umax(a, (unsigned int)b);
}

static inline __attribute__((host)) __attribute__((device)) long int max(long int a, long int b)
{




  if (sizeof(long int) == sizeof(int)) {



    return (long int)max((int)a, (int)b);
  } else {
    return (long int)llmax((long long int)a, (long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(unsigned long int a, unsigned long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umax((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmax((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(long int a, unsigned long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umax((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmax((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(unsigned long int a, long int b)
{



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    return (unsigned long int)umax((unsigned int)a, (unsigned int)b);
  } else {
    return (unsigned long int)ullmax((unsigned long long int)a, (unsigned long long int)b);
  }
}

static inline __attribute__((host)) __attribute__((device)) long long int max(long long int a, long long int b)
{
  return llmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(unsigned long long int a, unsigned long long int b)
{
  return ullmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(long long int a, unsigned long long int b)
{
  return ullmax((unsigned long long int)a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(unsigned long long int a, long long int b)
{
  return ullmax(a, (unsigned long long int)b);
}

static inline __attribute__((host)) __attribute__((device)) float max(float a, float b)
{
  return fmaxf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double max(double a, double b)
{
  return fmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double max(float a, double b)
{
  return fmax((double)a, b);
}

static inline __attribute__((host)) __attribute__((device)) double max(double a, float b)
{
  return fmax(a, (double)b);
}
# 10223 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/math_functions.h" 2
# 258 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/common_functions.h" 2
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/common_functions.h" 2
# 116 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_surface_types.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_surface_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_surface_types.h" 2






# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_surface_types.h" 2
# 77 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_surface_types.h"
template<class T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface : public surfaceReference
{

  __attribute__((host)) surface(void)
  {
    channelDesc = cudaCreateChannelDesc<T>();
  }

  __attribute__((host)) surface(struct cudaChannelFormatDesc desc)
  {
    channelDesc = desc;
  }

};

template<int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim> : public surfaceReference
{

  __attribute__((host)) surface(void)
  {
    channelDesc = cudaCreateChannelDesc<void>();
  }

};
# 117 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_texture_types.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_texture_types.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_texture_types.h" 2






# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_texture_types.h" 2
# 77 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_texture_types.h"
template<class T, int texType = 0x01, enum cudaTextureReadMode mode = cudaReadModeElementType>
struct __attribute__((device_builtin_texture_type)) texture : public textureReference
{

  __attribute__((host)) texture(int norm = 0,
                   enum cudaTextureFilterMode fMode = cudaFilterModePoint,
                   enum cudaTextureAddressMode aMode = cudaAddressModeClamp)
  {
    normalized = norm;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = cudaCreateChannelDesc<T>();
    sRGB = 0;
  }

  __attribute__((host)) texture(int norm,
                   enum cudaTextureFilterMode fMode,
                   enum cudaTextureAddressMode aMode,
                   struct cudaChannelFormatDesc desc)
  {
    normalized = norm;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = desc;
    sRGB = 0;
  }

};
# 118 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_functions.h" 1
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_types.h" 1
# 71 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 79 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
extern "C"
{
# 90 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __mulhi(int x, int y);
# 100 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __umulhi(unsigned int x, unsigned int y);
# 110 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __mul64hi(long long int x, long long int y);
# 120 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
# 129 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int_as_float(int x);
# 138 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float_as_int(float x);
# 147 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint_as_float(unsigned int x);
# 156 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float_as_uint(float x);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __syncthreads(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __prof_trigger(int);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __threadfence(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __threadfence_block(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __trap(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __brkpt(int c = 0);
# 185 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __saturatef(float x);
# 254 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __sad(int x, int y, unsigned int z);
# 322 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);
# 332 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __mul24(int x, int y);
# 342 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __umul24(unsigned int x, unsigned int y);
# 355 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float fdividef(float x, float y);
# 430 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdividef(float x, float y);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) double fdivide(double x, double y);
# 443 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __sinf(float x) throw ();
# 455 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __cosf(float x) throw ();
# 469 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __tanf(float x) throw ();
# 484 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void __sincosf(float x, float *sptr, float *cptr) throw ();
# 534 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __expf(float x) throw ();
# 566 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __exp10f(float x) throw ();
# 592 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __log2f(float x) throw ();
# 620 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __log10f(float x) throw ();
# 664 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __logf(float x) throw ();
# 707 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __powf(float x, float y) throw ();
# 716 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rn(float x);
# 725 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rz(float x);
# 734 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_ru(float);
# 743 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rd(float x);
# 752 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rn(float x);
# 761 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rz(float x);
# 770 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_ru(float x);
# 779 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rd(float x);
# 788 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rn(int x);
# 797 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rz(int x);
# 806 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_ru(int x);
# 815 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rd(int x);
# 824 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rn(unsigned int x);
# 833 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rz(unsigned int x);
# 842 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_ru(unsigned int x);
# 851 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rd(unsigned int x);
# 860 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rn(float x);
# 869 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rz(float x);
# 878 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_ru(float x);
# 887 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rd(float x);
# 896 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rn(float x);
# 905 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rz(float x);
# 914 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_ru(float x);
# 923 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rd(float x);
# 932 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rn(long long int x);
# 941 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rz(long long int x);
# 950 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_ru(long long int x);
# 959 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rd(long long int x);
# 968 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rn(unsigned long long int x);
# 977 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rz(unsigned long long int x);
# 986 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_ru(unsigned long long int x);
# 995 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rd(unsigned long long int x);
# 1007 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rn(float x, float y);
# 1019 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rz(float x, float y);
# 1031 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_ru(float x, float y);
# 1043 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rd(float x, float y);
# 1055 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rn(float x, float y);
# 1067 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rz(float x, float y);
# 1079 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_ru(float x, float y);
# 1091 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rd(float x, float y);
# 1103 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rn(float x, float y);
# 1115 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rz(float x, float y);
# 1127 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_ru(float x, float y);
# 1139 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rd(float x, float y);
# 1292 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rn(float x, float y, float z);
# 1445 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rz(float x, float y, float z);
# 1598 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_ru(float x, float y, float z);
# 1751 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rd(float x, float y, float z);
# 1784 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rn(float x);
# 1817 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rz(float x);
# 1850 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_ru(float x);
# 1883 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rd(float x);
# 1914 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rn(float x);
# 1945 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rz(float x);
# 1976 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_ru(float x);
# 2007 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rd(float x);
# 2046 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frsqrt_rn(float x);
# 2057 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rn(float x, float y);
# 2068 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rz(float x, float y);
# 2079 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_ru(float x, float y);
# 2090 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rd(float x, float y);
# 2099 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __clz(int x);
# 2110 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __ffs(int x);
# 2119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __popc(unsigned int x);
# 2128 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __brev(unsigned int x);
# 2137 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __clzll(long long int x);
# 2148 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __ffsll(long long int x);
# 2159 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __popcll(unsigned long long int x);
# 2168 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __brevll(unsigned long long int x);
# 2192 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
# 2204 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __hadd(int, int);
# 2217 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __rhadd(int, int);
# 2229 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __uhadd(unsigned int, unsigned int);
# 2242 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __urhadd(unsigned int, unsigned int);
# 2252 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __double2int_rz(double);
# 2261 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __double2uint_rz(double);
# 2270 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __double2ll_rz(double);
# 2279 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __double2ull_rz(double);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm0(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm1(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm2(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm3(void);
# 2309 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabs2(unsigned int a);
# 2320 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsss2(unsigned int a);
# 2331 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vadd2(unsigned int a, unsigned int b);
# 2342 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddss2 (unsigned int a, unsigned int b);
# 2352 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddus2 (unsigned int a, unsigned int b);
# 2363 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgs2(unsigned int a, unsigned int b);
# 2374 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgu2(unsigned int a, unsigned int b);
# 2385 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vhaddu2(unsigned int a, unsigned int b);
# 2396 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpeq2(unsigned int a, unsigned int b);
# 2407 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpges2(unsigned int a, unsigned int b);
# 2418 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgeu2(unsigned int a, unsigned int b);
# 2429 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgts2(unsigned int a, unsigned int b);
# 2440 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgtu2(unsigned int a, unsigned int b);
# 2451 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmples2(unsigned int a, unsigned int b);
# 2463 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpleu2(unsigned int a, unsigned int b);
# 2474 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmplts2(unsigned int a, unsigned int b);
# 2485 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpltu2(unsigned int a, unsigned int b);
# 2496 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpne2(unsigned int a, unsigned int b);
# 2507 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffu2(unsigned int a, unsigned int b);
# 2518 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxs2(unsigned int a, unsigned int b);
# 2529 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxu2(unsigned int a, unsigned int b);
# 2540 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmins2(unsigned int a, unsigned int b);
# 2551 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vminu2(unsigned int a, unsigned int b);
# 2562 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vseteq2(unsigned int a, unsigned int b);
# 2573 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetges2(unsigned int a, unsigned int b);
# 2584 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgeu2(unsigned int a, unsigned int b);
# 2595 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgts2(unsigned int a, unsigned int b);
# 2606 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgtu2(unsigned int a, unsigned int b);
# 2617 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetles2(unsigned int a, unsigned int b);
# 2628 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetleu2(unsigned int a, unsigned int b);
# 2639 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetlts2(unsigned int a, unsigned int b);
# 2650 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetltu2(unsigned int a, unsigned int b);
# 2661 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetne2(unsigned int a, unsigned int b);
# 2672 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsadu2(unsigned int a, unsigned int b);
# 2683 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsub2(unsigned int a, unsigned int b);
# 2694 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubss2 (unsigned int a, unsigned int b);
# 2705 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubus2 (unsigned int a, unsigned int b);
# 2715 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vneg2(unsigned int a);
# 2725 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vnegss2(unsigned int a);
# 2736 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffs2(unsigned int a, unsigned int b);
# 2747 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsads2(unsigned int a, unsigned int b);
# 2757 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabs4(unsigned int a);
# 2768 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsss4(unsigned int a);
# 2779 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vadd4(unsigned int a, unsigned int b);
# 2790 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddss4 (unsigned int a, unsigned int b);
# 2800 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddus4 (unsigned int a, unsigned int b);
# 2811 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgs4(unsigned int a, unsigned int b);
# 2822 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgu4(unsigned int a, unsigned int b);
# 2833 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vhaddu4(unsigned int a, unsigned int b);
# 2844 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpeq4(unsigned int a, unsigned int b);
# 2855 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpges4(unsigned int a, unsigned int b);
# 2866 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgeu4(unsigned int a, unsigned int b);
# 2877 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgts4(unsigned int a, unsigned int b);
# 2888 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgtu4(unsigned int a, unsigned int b);
# 2899 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmples4(unsigned int a, unsigned int b);
# 2910 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpleu4(unsigned int a, unsigned int b);
# 2921 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmplts4(unsigned int a, unsigned int b);
# 2932 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpltu4(unsigned int a, unsigned int b);
# 2943 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpne4(unsigned int a, unsigned int b);
# 2954 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffu4(unsigned int a, unsigned int b);
# 2965 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxs4(unsigned int a, unsigned int b);
# 2976 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxu4(unsigned int a, unsigned int b);
# 2987 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmins4(unsigned int a, unsigned int b);
# 2998 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vminu4(unsigned int a, unsigned int b);
# 3009 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vseteq4(unsigned int a, unsigned int b);
# 3020 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetles4(unsigned int a, unsigned int b);
# 3031 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetleu4(unsigned int a, unsigned int b);
# 3042 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetlts4(unsigned int a, unsigned int b);
# 3053 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetltu4(unsigned int a, unsigned int b);
# 3064 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetges4(unsigned int a, unsigned int b);
# 3075 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgeu4(unsigned int a, unsigned int b);
# 3086 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgts4(unsigned int a, unsigned int b);
# 3097 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgtu4(unsigned int a, unsigned int b);
# 3108 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetne4(unsigned int a, unsigned int b);
# 3119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsadu4(unsigned int a, unsigned int b);
# 3130 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsub4(unsigned int a, unsigned int b);
# 3141 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubss4(unsigned int a, unsigned int b);
# 3152 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubus4(unsigned int a, unsigned int b);
# 3162 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vneg4(unsigned int a);
# 3172 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vnegss4(unsigned int a);
# 3183 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffs4(unsigned int a, unsigned int b);
# 3194 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsads4(unsigned int a, unsigned int b);






}







static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int mulhi(int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(unsigned int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(unsigned int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) long long int mul64hi(long long int a, long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(long long int a, unsigned long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(unsigned long long int a, long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int float_as_int(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float int_as_float(int a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int float_as_uint(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float uint_as_float(unsigned int a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float saturate(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int mul24(int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int umul24(unsigned int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int float2int(float a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int float2uint(float a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float int2float(int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float uint2float(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest);
# 3259 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.hpp" 2
# 80 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.hpp"
static __inline__ __attribute__((device)) int mulhi(int a, int b)
{
  return __mulhi(a, b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(int a, unsigned int b)
{
  return __umulhi((unsigned int)a, b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(unsigned int a, int b)
{
  return __umulhi(a, (unsigned int)b);
}

static __inline__ __attribute__((device)) long long int mul64hi(long long int a, long long int b)
{
  return __mul64hi(a, b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b)
{
  return __umul64hi(a, b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(long long int a, unsigned long long int b)
{
  return __umul64hi((unsigned long long int)a, b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(unsigned long long int a, long long int b)
{
  return __umul64hi(a, (unsigned long long int)b);
}

static __inline__ __attribute__((device)) int float_as_int(float a)
{
  return __float_as_int(a);
}

static __inline__ __attribute__((device)) float int_as_float(int a)
{
  return __int_as_float(a);
}

static __inline__ __attribute__((device)) unsigned int float_as_uint(float a)
{
  return __float_as_uint(a);
}

static __inline__ __attribute__((device)) float uint_as_float(unsigned int a)
{
  return __uint_as_float(a);
}
static __inline__ __attribute__((device)) float saturate(float a)
{
  return __saturatef(a);
}

static __inline__ __attribute__((device)) int mul24(int a, int b)
{
  return __mul24(a, b);
}

static __inline__ __attribute__((device)) unsigned int umul24(unsigned int a, unsigned int b)
{
  return __umul24(a, b);
}

static __inline__ __attribute__((device)) int float2int(float a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __float2int_rn(a) :
         mode == cudaRoundPosInf ? __float2int_ru(a) :
         mode == cudaRoundMinInf ? __float2int_rd(a) :
                                    __float2int_rz(a);
}

static __inline__ __attribute__((device)) unsigned int float2uint(float a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __float2uint_rn(a) :
         mode == cudaRoundPosInf ? __float2uint_ru(a) :
         mode == cudaRoundMinInf ? __float2uint_rd(a) :
                                    __float2uint_rz(a);
}

static __inline__ __attribute__((device)) float int2float(int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __int2float_rz(a) :
         mode == cudaRoundPosInf ? __int2float_ru(a) :
         mode == cudaRoundMinInf ? __int2float_rd(a) :
                                   __int2float_rn(a);
}

static __inline__ __attribute__((device)) float uint2float(unsigned int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __uint2float_rz(a) :
         mode == cudaRoundPosInf ? __uint2float_ru(a) :
         mode == cudaRoundMinInf ? __uint2float_rd(a) :
                                   __uint2float_rn(a);
}
# 3260 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2


# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 2
# 77 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicAdd(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicAdd(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicExch(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicExch(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) float __fAtomicExch(float *address, float val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicMin(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicMin(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicMax(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicMax(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicDec(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicAnd(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicAnd(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicOr(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicOr(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicXor(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicXor(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicCAS(int *address, int compare, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val);
}
# 107 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
static __inline__ __attribute__((device)) int atomicAdd(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicAdd(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicSub(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicSub(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicExch(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicExch(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) float atomicExch(float *address, float val) ;

static __inline__ __attribute__((device)) int atomicMin(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicMin(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicMax(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicMax(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) unsigned int atomicInc(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) unsigned int atomicDec(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicAnd(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicAnd(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicOr(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicOr(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicXor(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicXor(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicCAS(int *address, int compare, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) ;







# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 156 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 157 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 2
# 173 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
extern "C"
{

extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicAdd(unsigned long long int *address, unsigned long long int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicExch(unsigned long long int *address, unsigned long long int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);

extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__any""() is not valid on compute_70 and above, and should be replaced with ""__any""_sync()." "To continue using ""__any""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __any(int cond);
extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__all""() is not valid on compute_70 and above, and should be replaced with ""__all""_sync()." "To continue using ""__all""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __all(int cond);
}
# 191 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
static __inline__ __attribute__((device)) unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val) ;

static __inline__ __attribute__((device)) unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val) ;

static __inline__ __attribute__((device)) unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val) ;

static __inline__ __attribute__((device)) __attribute__((deprecated("__any""() is not valid on compute_70 and above, and should be replaced with ""__any""_sync()." "To continue using ""__any""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) bool any(bool cond) ;

static __inline__ __attribute__((device)) __attribute__((deprecated("__all""() is not valid on compute_70 and above, and should be replaced with ""__all""_sync()." "To continue using ""__all""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) bool all(bool cond) ;
# 210 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp" 2







static __inline__ __attribute__((device)) int atomicAdd(int *address, int val)
{
  return __iAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicAdd(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) int atomicSub(int *address, int val)
{
  return __iAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device)) unsigned int atomicSub(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device)) int atomicExch(int *address, int val)
{
  return __iAtomicExch(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicExch(unsigned int *address, unsigned int val)
{
  return __uAtomicExch(address, val);
}

static __inline__ __attribute__((device)) float atomicExch(float *address, float val)
{
  return __fAtomicExch(address, val);
}

static __inline__ __attribute__((device)) int atomicMin(int *address, int val)
{
  return __iAtomicMin(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicMin(unsigned int *address, unsigned int val)
{
  return __uAtomicMin(address, val);
}

static __inline__ __attribute__((device)) int atomicMax(int *address, int val)
{
  return __iAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicMax(unsigned int *address, unsigned int val)
{
  return __uAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicInc(unsigned int *address, unsigned int val)
{
  return __uAtomicInc(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicDec(unsigned int *address, unsigned int val)
{
  return __uAtomicDec(address, val);
}

static __inline__ __attribute__((device)) int atomicAnd(int *address, int val)
{
  return __iAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicAnd(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) int atomicOr(int *address, int val)
{
  return __iAtomicOr(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicOr(unsigned int *address, unsigned int val)
{
  return __uAtomicOr(address, val);
}

static __inline__ __attribute__((device)) int atomicXor(int *address, int val)
{
  return __iAtomicXor(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicXor(unsigned int *address, unsigned int val)
{
  return __uAtomicXor(address, val);
}

static __inline__ __attribute__((device)) int atomicCAS(int *address, int compare, int val)
{
  return __iAtomicCAS(address, compare, val);
}

static __inline__ __attribute__((device)) unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  return __uAtomicCAS(address, compare, val);
}







# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 188 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 189 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.hpp" 2







static __inline__ __attribute__((device)) unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicExch(address, val);
}

static __inline__ __attribute__((device)) unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val)
{
  return __ullAtomicCAS(address, compare, val);
}

static __inline__ __attribute__((device)) bool any(bool cond)
{
  return (bool)__any((int)cond);
}

static __inline__ __attribute__((device)) bool all(bool cond)
{
  return (bool)__all((int)cond);
}
# 211 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_atomic_functions.h" 2
# 3263 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h" 1
# 73 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 74 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h" 2



extern "C"
{
# 87 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double_as_longlong(double x);
# 96 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __longlong_as_double(long long int x);
# 253 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rn(double x, double y, double z);
# 410 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rz(double x, double y, double z);
# 567 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_ru(double x, double y, double z);
# 724 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rd(double x, double y, double z);
# 736 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rn(double x, double y);
# 748 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rz(double x, double y);
# 760 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_ru(double x, double y);
# 772 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rd(double x, double y);
# 784 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rn(double x, double y);
# 796 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rz(double x, double y);
# 808 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_ru(double x, double y);
# 820 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rd(double x, double y);
# 832 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rn(double x, double y);
# 844 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rz(double x, double y);
# 856 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_ru(double x, double y);
# 868 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rd(double x, double y);
# 877 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rn(double x);
# 886 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rz(double x);
# 895 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_ru(double x);
# 904 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rd(double x);
# 913 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rn(double x);
# 922 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_ru(double x);
# 931 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rd(double x);
# 940 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rn(double x);
# 949 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_ru(double x);
# 958 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rd(double x);
# 967 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rn(double x);
# 976 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_ru(double x);
# 985 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rd(double x);
# 994 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rn(double x);
# 1003 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_ru(double x);
# 1012 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rd(double x);







extern __attribute__((device)) __attribute__((device_builtin)) double __int2double_rn(int x);







extern __attribute__((device)) __attribute__((device_builtin)) double __uint2double_rn(unsigned int x);
# 1037 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rn(long long int x);
# 1046 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rz(long long int x);
# 1055 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_ru(long long int x);
# 1064 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rd(long long int x);
# 1073 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rn(unsigned long long int x);
# 1082 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rz(unsigned long long int x);
# 1091 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_ru(unsigned long long int x);
# 1100 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rd(unsigned long long int x);
# 1109 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2hiint(double x);
# 1118 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2loint(double x);
# 1128 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __hiloint2double(int hi, int lo);
}







static __inline__ __attribute__((device)) double fma(double a, double b, double c, enum cudaRoundMode mode);

static __inline__ __attribute__((device)) double dmul(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double dadd(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double dsub(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) int double2int(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) unsigned int double2uint(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) long long int double2ll(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) unsigned long long int double2ull(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) double ll2double(long long int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double ull2double(unsigned long long int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double int2double(int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double uint2double(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double float2double(float a, enum cudaRoundMode mode = cudaRoundNearest);






# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.hpp" 1
# 73 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 74 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.hpp" 2
# 83 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.hpp"
static __inline__ __attribute__((device)) double fma(double a, double b, double c, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __fma_rz(a, b, c) :
         mode == cudaRoundPosInf ? __fma_ru(a, b, c) :
         mode == cudaRoundMinInf ? __fma_rd(a, b, c) :
                                   __fma_rn(a, b, c);
}

static __inline__ __attribute__((device)) double dmul(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dmul_rz(a, b) :
         mode == cudaRoundPosInf ? __dmul_ru(a, b) :
         mode == cudaRoundMinInf ? __dmul_rd(a, b) :
                                   __dmul_rn(a, b);
}

static __inline__ __attribute__((device)) double dadd(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dadd_rz(a, b) :
         mode == cudaRoundPosInf ? __dadd_ru(a, b) :
         mode == cudaRoundMinInf ? __dadd_rd(a, b) :
                                   __dadd_rn(a, b);
}

static __inline__ __attribute__((device)) double dsub(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dsub_rz(a, b) :
         mode == cudaRoundPosInf ? __dsub_ru(a, b) :
         mode == cudaRoundMinInf ? __dsub_rd(a, b) :
                                   __dsub_rn(a, b);
}

static __inline__ __attribute__((device)) int double2int(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2int_rn(a) :
         mode == cudaRoundPosInf ? __double2int_ru(a) :
         mode == cudaRoundMinInf ? __double2int_rd(a) :
                                    __double2int_rz(a);
}

static __inline__ __attribute__((device)) unsigned int double2uint(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2uint_rn(a) :
         mode == cudaRoundPosInf ? __double2uint_ru(a) :
         mode == cudaRoundMinInf ? __double2uint_rd(a) :
                                    __double2uint_rz(a);
}

static __inline__ __attribute__((device)) long long int double2ll(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ll_rn(a) :
         mode == cudaRoundPosInf ? __double2ll_ru(a) :
         mode == cudaRoundMinInf ? __double2ll_rd(a) :
                                    __double2ll_rz(a);
}

static __inline__ __attribute__((device)) unsigned long long int double2ull(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ull_rn(a) :
         mode == cudaRoundPosInf ? __double2ull_ru(a) :
         mode == cudaRoundMinInf ? __double2ull_rd(a) :
                                    __double2ull_rz(a);
}

static __inline__ __attribute__((device)) double ll2double(long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __ll2double_rz(a) :
         mode == cudaRoundPosInf ? __ll2double_ru(a) :
         mode == cudaRoundMinInf ? __ll2double_rd(a) :
                                   __ll2double_rn(a);
}

static __inline__ __attribute__((device)) double ull2double(unsigned long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __ull2double_rz(a) :
         mode == cudaRoundPosInf ? __ull2double_ru(a) :
         mode == cudaRoundMinInf ? __ull2double_rd(a) :
                                   __ull2double_rn(a);
}

static __inline__ __attribute__((device)) double int2double(int a, enum cudaRoundMode mode)
{
  return (double)a;
}

static __inline__ __attribute__((device)) double uint2double(unsigned int a, enum cudaRoundMode mode)
{
  return (double)a;
}

static __inline__ __attribute__((device)) double float2double(float a, enum cudaRoundMode mode)
{
  return (double)a;
}
# 1169 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_double_functions.h" 2
# 3264 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h" 2
# 78 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) float __fAtomicAdd(float *address, float val);
}
# 90 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h"
static __inline__ __attribute__((device)) float atomicAdd(float *address, float val) ;







# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.hpp" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.hpp" 2







static __inline__ __attribute__((device)) float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}
# 99 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_atomic_functions.h" 2
# 3265 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 71 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h" 2
# 80 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) long long __illAtomicMin(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __illAtomicMax(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicAnd(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicOr(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicXor(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicMin(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicMax(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicAnd(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicOr (unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicXor(unsigned long long *address, unsigned long long val);
}
# 101 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h"
static __inline__ __attribute__((device)) long long atomicMin(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicMax(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicAnd(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicOr(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicXor(long long *address, long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicMin(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicMax(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicOr(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicXor(unsigned long long *address, unsigned long long val) ;
# 129 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 71 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.hpp" 2







static __inline__ __attribute__((device)) long long atomicMin(long long *address, long long val)
{
    return __illAtomicMin(address, val);
}

static __inline__ __attribute__((device)) long long atomicMax(long long *address, long long val)
{
    return __illAtomicMax(address, val);
}

static __inline__ __attribute__((device)) long long atomicAnd(long long *address, long long val)
{
    return __llAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) long long atomicOr(long long *address, long long val)
{
    return __llAtomicOr(address, val);
}

static __inline__ __attribute__((device)) long long atomicXor(long long *address, long long val)
{
    return __llAtomicXor(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicMin(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicMin(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicMax(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicAnd(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicOr(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicOr(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicXor(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicXor(address, val);
}
# 130 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h" 2
# 3266 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_atomic_functions.h" 1
# 56 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_atomic_functions.h" 1
# 57 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_atomic_functions.h" 2
# 3267 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 71 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h" 2
# 82 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) double __dAtomicAdd(double *address, double val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicAdd_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicAdd_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicAdd_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicAdd_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicAdd_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicAdd_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
float __fAtomicAdd_block(float *address, float val);

extern __attribute__((device)) __attribute__((device_builtin))
float __fAtomicAdd_system(float *address, float val);

extern __attribute__((device)) __attribute__((device_builtin))
double __dAtomicAdd_block(double *address, double val);

extern __attribute__((device)) __attribute__((device_builtin))
double __dAtomicAdd_system(double *address, double val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicExch_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicExch_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicExch_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicExch_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicExch_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicExch_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
float __fAtomicExch_block(float *address, float val);

extern __attribute__((device)) __attribute__((device_builtin))
float __fAtomicExch_system(float *address, float val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicMin_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicMin_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __illAtomicMin_block(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __illAtomicMin_system(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicMin_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicMin_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicMin_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicMin_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicMax_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicMax_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __illAtomicMax_block(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __illAtomicMax_system(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicMax_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicMax_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicMax_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicMax_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicInc_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicInc_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicDec_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicDec_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicCAS_block(int *address, int compare, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicCAS_system(int *address, int compare, int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicCAS_block(unsigned int *address, unsigned int compare,
                                unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicCAS_system(unsigned int *address, unsigned int compare,
                                 unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicCAS_block(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicCAS_system(unsigned long long int *address,
                                         unsigned long long int compare,
                                         unsigned long long int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicAnd_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicAnd_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicAnd_block(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicAnd_system(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicAnd_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicAnd_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicAnd_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicAnd_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicOr_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicOr_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicOr_block(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicOr_system(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicOr_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicOr_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicOr_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicOr_system(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicXor_block(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
int __iAtomicXor_system(int *address, int val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicXor_block(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
long long __llAtomicXor_system(long long *address, long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicXor_block(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned int __uAtomicXor_system(unsigned int *address, unsigned int val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicXor_block(unsigned long long *address, unsigned long long val);

extern __attribute__((device)) __attribute__((device_builtin))
unsigned long long __ullAtomicXor_system(unsigned long long *address, unsigned long long val);
}
# 304 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h"
static __inline__ __attribute__((device)) double atomicAdd(double *address, double val) ;

static __inline__ __attribute__((device))
int atomicAdd_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicAdd_system(int *address, int val) ;

static __inline__ __attribute__((device))
unsigned int atomicAdd_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicAdd_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
float atomicAdd_block(float *address, float val) ;

static __inline__ __attribute__((device))
float atomicAdd_system(float *address, float val) ;

static __inline__ __attribute__((device))
double atomicAdd_block(double *address, double val) ;

static __inline__ __attribute__((device))
double atomicAdd_system(double *address, double val) ;

static __inline__ __attribute__((device))
int atomicSub_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicSub_system(int *address, int val) ;

static __inline__ __attribute__((device))
unsigned int atomicSub_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicSub_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
int atomicExch_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicExch_system(int *address, int val) ;

static __inline__ __attribute__((device))
unsigned int atomicExch_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicExch_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
float atomicExch_block(float *address, float val) ;

static __inline__ __attribute__((device))
float atomicExch_system(float *address, float val) ;

static __inline__ __attribute__((device))
int atomicMin_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicMin_system(int *address, int val) ;

static __inline__ __attribute__((device))
long long atomicMin_block(long long *address, long long val) ;

static __inline__ __attribute__((device))
long long atomicMin_system(long long *address, long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicMin_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicMin_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
int atomicMax_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicMax_system(int *address, int val) ;

static __inline__ __attribute__((device))
long long atomicMax_block(long long *address, long long val) ;

static __inline__ __attribute__((device))
long long atomicMax_system(long long *address, long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicMax_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicMax_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicInc_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicInc_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicDec_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicDec_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
int atomicCAS_block(int *address, int compare, int val) ;

static __inline__ __attribute__((device))
int atomicCAS_system(int *address, int compare, int val) ;

static __inline__ __attribute__((device))
unsigned int atomicCAS_block(unsigned int *address, unsigned int compare,
                             unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicCAS_system(unsigned int *address, unsigned int compare,
                              unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long int atomicCAS_block(unsigned long long int *address,
                                       unsigned long long int compare,
                                       unsigned long long int val) ;

static __inline__ __attribute__((device))
unsigned long long int atomicCAS_system(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val) ;

static __inline__ __attribute__((device))
int atomicAnd_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicAnd_system(int *address, int val) ;

static __inline__ __attribute__((device))
long long atomicAnd_block(long long *address, long long val) ;

static __inline__ __attribute__((device))
long long atomicAnd_system(long long *address, long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicAnd_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicAnd_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
int atomicOr_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicOr_system(int *address, int val) ;

static __inline__ __attribute__((device))
long long atomicOr_block(long long *address, long long val) ;

static __inline__ __attribute__((device))
long long atomicOr_system(long long *address, long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicOr_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicOr_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
int atomicXor_block(int *address, int val) ;

static __inline__ __attribute__((device))
int atomicXor_system(int *address, int val) ;

static __inline__ __attribute__((device))
long long atomicXor_block(long long *address, long long val) ;

static __inline__ __attribute__((device))
long long atomicXor_system(long long *address, long long val) ;

static __inline__ __attribute__((device))
unsigned int atomicXor_block(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned int atomicXor_system(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device))
unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device))
unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) ;
# 536 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.hpp" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 71 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.hpp" 2







static __inline__ __attribute__((device)) double atomicAdd(double *address, double val)
{
  return __dAtomicAdd(address, val);
}

static __inline__ __attribute__((device))
int atomicAdd_block(int *address, int val)
{
  return __iAtomicAdd_block(address, val);
}

static __inline__ __attribute__((device))
int atomicAdd_system(int *address, int val)
{
  return __iAtomicAdd_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicAdd_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicAdd_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAdd_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAdd_system(address, val);
}

static __inline__ __attribute__((device))
float atomicAdd_block(float *address, float val)
{
  return __fAtomicAdd_block(address, val);
}

static __inline__ __attribute__((device))
float atomicAdd_system(float *address, float val)
{
  return __fAtomicAdd_system(address, val);
}

static __inline__ __attribute__((device))
double atomicAdd_block(double *address, double val)
{
  return __dAtomicAdd_block(address, val);
}

static __inline__ __attribute__((device))
double atomicAdd_system(double *address, double val)
{
  return __dAtomicAdd_system(address, val);
}

static __inline__ __attribute__((device))
int atomicSub_block(int *address, int val)
{
  return __iAtomicAdd_block(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device))
int atomicSub_system(int *address, int val)
{
  return __iAtomicAdd_system(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device))
unsigned int atomicSub_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_block(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device))
unsigned int atomicSub_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_system(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device))
int atomicExch_block(int *address, int val)
{
  return __iAtomicExch_block(address, val);
}

static __inline__ __attribute__((device))
int atomicExch_system(int *address, int val)
{
  return __iAtomicExch_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicExch_block(unsigned int *address, unsigned int val)
{
  return __uAtomicExch_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicExch_system(unsigned int *address, unsigned int val)
{
  return __uAtomicExch_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicExch_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicExch_system(address, val);
}

static __inline__ __attribute__((device))
float atomicExch_block(float *address, float val)
{
  return __fAtomicExch_block(address, val);
}

static __inline__ __attribute__((device))
float atomicExch_system(float *address, float val)
{
  return __fAtomicExch_system(address, val);
}

static __inline__ __attribute__((device))
int atomicMin_block(int *address, int val)
{
  return __iAtomicMin_block(address, val);
}

static __inline__ __attribute__((device))
int atomicMin_system(int *address, int val)
{
  return __iAtomicMin_system(address, val);
}

static __inline__ __attribute__((device))
long long atomicMin_block(long long *address, long long val)
{
  return __illAtomicMin_block(address, val);
}

static __inline__ __attribute__((device))
long long atomicMin_system(long long *address, long long val)
{
  return __illAtomicMin_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicMin_block(unsigned int *address, unsigned int val)
{
  return __uAtomicMin_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicMin_system(unsigned int *address, unsigned int val)
{
  return __uAtomicMin_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMin_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMin_system(address, val);
}

static __inline__ __attribute__((device))
int atomicMax_block(int *address, int val)
{
  return __iAtomicMax_block(address, val);
}

static __inline__ __attribute__((device))
int atomicMax_system(int *address, int val)
{
  return __iAtomicMax_system(address, val);
}

static __inline__ __attribute__((device))
long long atomicMax_block(long long *address, long long val)
{
  return __illAtomicMax_block(address, val);
}

static __inline__ __attribute__((device))
long long atomicMax_system(long long *address, long long val)
{
  return __illAtomicMax_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicMax_block(unsigned int *address, unsigned int val)
{
  return __uAtomicMax_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicMax_system(unsigned int *address, unsigned int val)
{
  return __uAtomicMax_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMax_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMax_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicInc_block(unsigned int *address, unsigned int val)
{
  return __uAtomicInc_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicInc_system(unsigned int *address, unsigned int val)
{
  return __uAtomicInc_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicDec_block(unsigned int *address, unsigned int val)
{
  return __uAtomicDec_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicDec_system(unsigned int *address, unsigned int val)
{
  return __uAtomicDec_system(address, val);
}

static __inline__ __attribute__((device))
int atomicCAS_block(int *address, int compare, int val)
{
  return __iAtomicCAS_block(address, compare, val);
}

static __inline__ __attribute__((device))
int atomicCAS_system(int *address, int compare, int val)
{
  return __iAtomicCAS_system(address, compare, val);
}

static __inline__ __attribute__((device))
unsigned int atomicCAS_block(unsigned int *address, unsigned int compare,
                             unsigned int val)
{
  return __uAtomicCAS_block(address, compare, val);
}

static __inline__ __attribute__((device))
unsigned int atomicCAS_system(unsigned int *address, unsigned int compare,
                              unsigned int val)
{
  return __uAtomicCAS_system(address, compare, val);
}

static __inline__ __attribute__((device))
unsigned long long int atomicCAS_block(unsigned long long int *address,
                                       unsigned long long int compare,
                                       unsigned long long int val)
{
  return __ullAtomicCAS_block(address, compare, val);
}

static __inline__ __attribute__((device))
unsigned long long int atomicCAS_system(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val)
{
  return __ullAtomicCAS_system(address, compare, val);
}

static __inline__ __attribute__((device))
int atomicAnd_block(int *address, int val)
{
  return __iAtomicAnd_block(address, val);
}

static __inline__ __attribute__((device))
int atomicAnd_system(int *address, int val)
{
  return __iAtomicAnd_system(address, val);
}

static __inline__ __attribute__((device))
long long atomicAnd_block(long long *address, long long val)
{
  return __llAtomicAnd_block(address, val);
}

static __inline__ __attribute__((device))
long long atomicAnd_system(long long *address, long long val)
{
  return __llAtomicAnd_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicAnd_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicAnd_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAnd_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAnd_system(address, val);
}

static __inline__ __attribute__((device))
int atomicOr_block(int *address, int val)
{
  return __iAtomicOr_block(address, val);
}

static __inline__ __attribute__((device))
int atomicOr_system(int *address, int val)
{
  return __iAtomicOr_system(address, val);
}

static __inline__ __attribute__((device))
long long atomicOr_block(long long *address, long long val)
{
  return __llAtomicOr_block(address, val);
}

static __inline__ __attribute__((device))
long long atomicOr_system(long long *address, long long val)
{
  return __llAtomicOr_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicOr_block(unsigned int *address, unsigned int val)
{
  return __uAtomicOr_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicOr_system(unsigned int *address, unsigned int val)
{
  return __uAtomicOr_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicOr_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicOr_system(address, val);
}

static __inline__ __attribute__((device))
int atomicXor_block(int *address, int val)
{
  return __iAtomicXor_block(address, val);
}

static __inline__ __attribute__((device))
int atomicXor_system(int *address, int val)
{
  return __iAtomicXor_system(address, val);
}

static __inline__ __attribute__((device))
long long atomicXor_block(long long *address, long long val)
{
  return __llAtomicXor_block(address, val);
}

static __inline__ __attribute__((device))
long long atomicXor_system(long long *address, long long val)
{
  return __llAtomicXor_system(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicXor_block(unsigned int *address, unsigned int val)
{
  return __uAtomicXor_block(address, val);
}

static __inline__ __attribute__((device))
unsigned int atomicXor_system(unsigned int *address, unsigned int val)
{
  return __uAtomicXor_system(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicXor_block(address, val);
}

static __inline__ __attribute__((device))
unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicXor_system(address, val);
}
# 537 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_60_atomic_functions.h" 2
# 3268 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h" 2
# 92 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) void __threadfence_system(void);
# 106 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rn(double x, double y);
# 118 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rz(double x, double y);
# 130 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_ru(double x, double y);
# 142 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rd(double x, double y);
# 176 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rn(double x);
# 210 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rz(double x);
# 244 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_ru(double x);
# 278 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rd(double x);
# 310 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rn(double x);
# 342 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rz(double x);
# 374 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_ru(double x);
# 406 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rd(double x);
extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__ballot""() is not valid on compute_70 and above, and should be replaced with ""__ballot""_sync()." "To continue using ""__ballot""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int __ballot(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_count(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_and(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_or(int);
extern __attribute__((device)) __attribute__((device_builtin)) long long int clock64(void);






extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rn(float, float, float);
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rz(float, float, float);
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_ru(float, float, float);
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rd(float, float, float);
# 433 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double_as_longlong(double x);
# 442 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __longlong_as_double(long long int x);
# 599 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rn(double x, double y, double z);
# 756 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rz(double x, double y, double z);
# 913 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_ru(double x, double y, double z);
# 1070 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rd(double x, double y, double z);
# 1082 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rn(double x, double y);
# 1094 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rz(double x, double y);
# 1106 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_ru(double x, double y);
# 1118 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rd(double x, double y);
# 1130 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rn(double x, double y);
# 1142 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rz(double x, double y);
# 1154 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_ru(double x, double y);
# 1166 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rd(double x, double y);
# 1178 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rn(double x, double y);
# 1190 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rz(double x, double y);
# 1202 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_ru(double x, double y);
# 1214 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rd(double x, double y);
# 1223 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rn(double x);
# 1232 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rz(double x);
# 1241 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_ru(double x);
# 1250 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rd(double x);
# 1259 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rn(double x);
# 1268 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_ru(double x);
# 1277 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rd(double x);
# 1286 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rn(double x);
# 1295 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_ru(double x);
# 1304 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rd(double x);
# 1313 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rn(double x);
# 1322 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_ru(double x);
# 1331 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rd(double x);
# 1340 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rn(double x);
# 1349 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_ru(double x);
# 1358 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rd(double x);







extern __attribute__((device)) __attribute__((device_builtin)) double __int2double_rn(int x);







extern __attribute__((device)) __attribute__((device_builtin)) double __uint2double_rn(unsigned int x);
# 1383 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rn(long long int x);
# 1392 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rz(long long int x);
# 1401 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_ru(long long int x);
# 1410 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rd(long long int x);
# 1419 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rn(unsigned long long int x);
# 1428 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rz(unsigned long long int x);
# 1437 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_ru(unsigned long long int x);
# 1446 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rd(unsigned long long int x);
# 1455 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2hiint(double x);
# 1464 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2loint(double x);
# 1474 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __hiloint2double(int hi, int lo);


}






static __inline__ __attribute__((device)) __attribute__((deprecated("__ballot""() is not valid on compute_70 and above, and should be replaced with ""__ballot""_sync()." "To continue using ""__ballot""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int ballot(bool pred) ;

static __inline__ __attribute__((device)) int syncthreads_count(bool pred) ;

static __inline__ __attribute__((device)) bool syncthreads_and(bool pred) ;

static __inline__ __attribute__((device)) bool syncthreads_or(bool pred) ;






static __inline__ __attribute__((device)) unsigned int __isGlobal(const void *ptr) ;







# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.hpp" 1
# 67 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.hpp" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.hpp" 2







static __inline__ __attribute__((device)) unsigned int ballot(bool pred)
{
  return __ballot((int)pred);
}

static __inline__ __attribute__((device)) int syncthreads_count(bool pred)
{
  return __syncthreads_count((int)pred);
}

static __inline__ __attribute__((device)) bool syncthreads_and(bool pred)
{
  return (bool)__syncthreads_and((int)pred);
}

static __inline__ __attribute__((device)) bool syncthreads_or(bool pred)
{
  return (bool)__syncthreads_or((int)pred);
}




static __inline__ __attribute__((device)) unsigned int __isGlobal(const void *ptr)
{
    unsigned int ret;
    asm volatile ("{ \n\t"
                  "    .reg .pred p; \n\t"
                  "    isspacep.global p, %1; \n\t"
                  "    selp.u32 %0, 1, 0, p;  \n\t"

                  "} \n\t" : "=r"(ret) : "l"(ptr));




    return ret;
}
# 1506 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_20_intrinsics.h" 2
# 3269 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h" 2
# 107 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h"
static __attribute__((device)) __inline__ unsigned __fns(unsigned mask, unsigned base, int offset) ;
static __attribute__((device)) __inline__ void __barrier_sync(unsigned id) ;
static __attribute__((device)) __inline__ void __barrier_sync_count(unsigned id, unsigned cnt) ;
static __attribute__((device)) __inline__ void __syncwarp(unsigned mask=0xFFFFFFFF) ;
static __attribute__((device)) __inline__ int __all_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ int __any_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ int __uni_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ unsigned __ballot_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ unsigned __activemask() ;







static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __shfl(int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_sync(unsigned mask, int var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int __shfl(unsigned int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_sync(unsigned mask, unsigned int var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __shfl_up(int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int __shfl_up(unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_up_sync(unsigned mask, unsigned int var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __shfl_down(int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int __shfl_down(unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_down_sync(unsigned mask, unsigned int var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) int __shfl_xor(int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned int __shfl_xor(unsigned int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_xor_sync(unsigned mask, unsigned int var, int laneMask, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) float __shfl(float var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_sync(unsigned mask, float var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) float __shfl_up(float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) float __shfl_down(float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) float __shfl_xor(float var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width=32) ;


static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long long __shfl(long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_sync(unsigned mask, long long var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long long __shfl(unsigned long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long long __shfl_up(long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_up_sync(unsigned mask, long long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long long __shfl_down(long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_down_sync(unsigned mask, long long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long long __shfl_xor(long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) double __shfl(double var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_sync(unsigned mask, double var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) double __shfl_up(double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_up_sync(unsigned mask, double var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) double __shfl_down(double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_down_sync(unsigned mask, double var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) double __shfl_xor(double var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width=32) ;



static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long __shfl(long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_sync(unsigned mask, long var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is not valid on compute_70 and above, and should be replaced with ""__shfl""_sync()." "To continue using ""__shfl""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long __shfl(unsigned long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long __shfl_up(long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_up_sync(unsigned mask, long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is not valid on compute_70 and above, and should be replaced with ""__shfl_up""_sync()." "To continue using ""__shfl_up""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long __shfl_up(unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long __shfl_down(long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_down_sync(unsigned mask, long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is not valid on compute_70 and above, and should be replaced with ""__shfl_down""_sync()." "To continue using ""__shfl_down""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long __shfl_down(unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned int delta, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) long __shfl_xor(long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width=32) ;

static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is not valid on compute_70 and above, and should be replaced with ""__shfl_xor""_sync()." "To continue using ""__shfl_xor""(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."))) unsigned long __shfl_xor(unsigned long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width=32) ;
# 238 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.hpp" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.hpp" 2



extern "C"
{
}
# 91 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.hpp"
static __attribute__((device)) __inline__
unsigned __fns(unsigned mask, unsigned base, int offset) {
  extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __nvvm_fns(unsigned int mask, unsigned int base, int offset);
  return __nvvm_fns(mask, base, offset);
}

static __attribute__((device)) __inline__
void __barrier_sync(unsigned id) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_barrier_sync(unsigned id);
  return __nvvm_barrier_sync(id);
}

static __attribute__((device)) __inline__
void __barrier_sync_count(unsigned id, unsigned cnt) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_barrier_sync_cnt(unsigned id, unsigned cnt);
  return __nvvm_barrier_sync_cnt(id, cnt);
}

static __attribute__((device)) __inline__
void __syncwarp(unsigned mask) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_bar_warp_sync(unsigned mask);
  return __nvvm_bar_warp_sync(mask);
}

static __attribute__((device)) __inline__
int __all_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_all_sync(unsigned int mask, int pred);
  return __nvvm_vote_all_sync(mask, pred);
}

static __attribute__((device)) __inline__
int __any_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_any_sync(unsigned int mask, int pred);
  return __nvvm_vote_any_sync(mask, pred);
}

static __attribute__((device)) __inline__
int __uni_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_uni_sync(unsigned int mask, int pred);
  return __nvvm_vote_uni_sync(mask, pred);
}

static __attribute__((device)) __inline__
unsigned __ballot_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __nvvm_vote_ballot_sync(unsigned int mask, int pred);
  return __nvvm_vote_ballot_sync(mask, pred);
}

static __attribute__((device)) __inline__
unsigned __activemask() {
    unsigned ret;
    asm volatile ("activemask.b32 %0;" : "=r"(ret));
    return ret;
}






static __attribute__((device)) __inline__ int __shfl(int var, int srcLane, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(srcLane), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ int __shfl_sync(unsigned mask, int var, int srcLane, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, var, srcLane, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl(unsigned int var, int srcLane, int width) {
 return (unsigned int) __shfl((int)var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned int __shfl_sync(unsigned mask, unsigned int var, int srcLane, int width) {
        return (unsigned int) __shfl_sync(mask, (int)var, srcLane, width);
}

static __attribute__((device)) __inline__ int __shfl_up(int var, unsigned int delta, int width) {
 int ret;
 int c = (32 -width) << 8;
 asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = (32 -width) << 8;
        ret = __nvvm_shfl_up_sync(mask, var, delta, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_up(unsigned int var, unsigned int delta, int width) {
 return (unsigned int) __shfl_up((int)var, delta, width);
}

static __attribute__((device)) __inline__ unsigned int __shfl_up_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_up_sync(mask, (int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_down(int var, unsigned int delta, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, var, delta, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_down(unsigned int var, unsigned int delta, int width) {
 return (unsigned int) __shfl_down((int)var, delta, width);
}

static __attribute__((device)) __inline__ unsigned int __shfl_down_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_down_sync(mask, (int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_xor(int var, int laneMask, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(laneMask), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_xor(unsigned int var, int laneMask, int width) {
 return (unsigned int) __shfl_xor((int)var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned int __shfl_xor_sync(unsigned mask, unsigned int var, int laneMask, int width) {
 return (unsigned int) __shfl_xor_sync(mask, (int)var, laneMask, width);
}

static __attribute__((device)) __inline__ float __shfl(float var, int srcLane, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(srcLane), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_sync(unsigned mask, float var, int srcLane, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
        int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, __float_as_int(var), srcLane, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_up(float var, unsigned int delta, int width) {
 float ret;
        int c;
 c = (32 -width) << 8;
 asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = (32 -width) << 8;
        ret = __nvvm_shfl_up_sync(mask, __float_as_int(var), delta, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_down(float var, unsigned int delta, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, __float_as_int(var), delta, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_xor(float var, int laneMask, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, __float_as_int(var), laneMask, c);
 return __int_as_float(ret);
}


static __attribute__((device)) __inline__ long long __shfl(long long var, int srcLane, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl(hi, srcLane, width);
 lo = __shfl(lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ long long __shfl_sync(unsigned mask, long long var, int srcLane, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_sync(mask, hi, srcLane, width);
 lo = __shfl_sync(mask, lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl(unsigned long long var, int srcLane, int width) {
 return (unsigned long long) __shfl((long long) var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width) {
        return (unsigned long long) __shfl_sync(mask, (long long) var, srcLane, width);
}

static __attribute__((device)) __inline__ long long __shfl_up(long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_up(hi, delta, width);
 lo = __shfl_up(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ long long __shfl_up_sync(unsigned mask, long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_up_sync(mask, hi, delta, width);
 lo = __shfl_up_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width) {
 return (unsigned long long) __shfl_up((long long) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_up_sync(mask, (long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_down(long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_down(hi, delta, width);
 lo = __shfl_down(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ long long __shfl_down_sync(unsigned mask, long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_down_sync(mask, hi, delta, width);
 lo = __shfl_down_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width) {
 return (unsigned long long) __shfl_down((long long) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_down_sync(mask, (long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_xor(long long var, int laneMask, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_xor(hi, laneMask, width);
 lo = __shfl_xor(lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_xor_sync(mask, hi, laneMask, width);
 lo = __shfl_xor_sync(mask, lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width) {
 return (unsigned long long) __shfl_xor((long long) var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width) {
        return (unsigned long long) __shfl_xor_sync(mask, (long long) var, laneMask, width);
}

static __attribute__((device)) __inline__ double __shfl(double var, int srcLane, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl(hi, srcLane, width);
 lo = __shfl(lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_sync(unsigned mask, double var, int srcLane, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_sync(mask, hi, srcLane, width);
 lo = __shfl_sync(mask, lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}


static __attribute__((device)) __inline__ double __shfl_up(double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_up(hi, delta, width);
 lo = __shfl_up(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_up_sync(unsigned mask, double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_up_sync(mask, hi, delta, width);
 lo = __shfl_up_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_down(double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_down(hi, delta, width);
 lo = __shfl_down(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_down_sync(unsigned mask, double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_down_sync(mask, hi, delta, width);
 lo = __shfl_down_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_xor(double var, int laneMask, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_xor(hi, laneMask, width);
 lo = __shfl_xor(lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_xor_sync(mask, hi, laneMask, width);
 lo = __shfl_xor_sync(mask, lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}



static __attribute__((device)) __inline__ long __shfl(long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl((long long) var, srcLane, width) :
  __shfl((int) var, srcLane, width);
}

static __attribute__((device)) __inline__ long __shfl_sync(unsigned mask, long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (long long) var, srcLane, width) :
  __shfl_sync(mask, (int) var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl(unsigned long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl((unsigned long long) var, srcLane, width) :
  __shfl((unsigned int) var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (unsigned long long) var, srcLane, width) :
  __shfl_sync(mask, (unsigned int) var, srcLane, width);
}

static __attribute__((device)) __inline__ long __shfl_up(long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up((long long) var, delta, width) :
  __shfl_up((int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_up_sync(unsigned mask, long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up_sync(mask, (long long) var, delta, width) :
  __shfl_up_sync(mask, (int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_up(unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up((unsigned long long) var, delta, width) :
  __shfl_up((unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up_sync(mask, (unsigned long long) var, delta, width) :
  __shfl_up_sync(mask, (unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_down(long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down((long long) var, delta, width) :
  __shfl_down((int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_down_sync(unsigned mask, long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down_sync(mask, (long long) var, delta, width) :
  __shfl_down_sync(mask, (int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_down(unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down((unsigned long long) var, delta, width) :
  __shfl_down((unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down_sync(mask, (unsigned long long) var, delta, width) :
  __shfl_down_sync(mask, (unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_xor(long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor((long long) var, laneMask, width) :
  __shfl_xor((int) var, laneMask, width);
}

static __attribute__((device)) __inline__ long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor_sync(mask, (long long) var, laneMask, width) :
  __shfl_xor_sync(mask, (int) var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_xor(unsigned long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor((unsigned long long) var, laneMask, width) :
  __shfl_xor((unsigned int) var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor_sync(mask, (unsigned long long) var, laneMask, width) :
  __shfl_xor_sync(mask, (unsigned int) var, laneMask, width);
}
# 239 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_30_intrinsics.h" 2
# 3270 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h" 2
# 89 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ long __ldg(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldg(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldg(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldg(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldg(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldg(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldg(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldg(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldg(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldg(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldg(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldg(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldg(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldg(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldg(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldg(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldg(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldg(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldg(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldg(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldg(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldg(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldg(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldg(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldg(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldg(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldg(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldg(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldg(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldg(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldcg(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldcg(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldcg(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldcg(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldcg(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldcg(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldcg(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldcg(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldcg(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldcg(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldcg(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldcg(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldcg(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldcg(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldcg(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldcg(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldcg(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldcg(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldcg(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldcg(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldcg(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldcg(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldcg(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldcg(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldcg(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldcg(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldcg(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldcg(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldcg(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldcg(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldca(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldca(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldca(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldca(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldca(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldca(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldca(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldca(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldca(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldca(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldca(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldca(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldca(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldca(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldca(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldca(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldca(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldca(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldca(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldca(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldca(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldca(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldca(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldca(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldca(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldca(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldca(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldca(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldca(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldca(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldcs(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldcs(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldcs(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldcs(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldcs(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldcs(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldcs(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldcs(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldcs(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldcs(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldcs(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldcs(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldcs(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldcs(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldcs(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldcs(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldcs(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldcs(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldcs(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldcs(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldcs(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldcs(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldcs(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldcs(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldcs(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldcs(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldcs(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldcs(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldcs(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldcs(const double2 *ptr) ;
# 246 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 258 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 271 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 283 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 293 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp" 2



extern "C"
{


}
# 103 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ long __ldg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.nc.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.nc.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.nc.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.nc.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.nc.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.nc.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.nc.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.nc.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.nc.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.nc.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldg(const float *ptr) { float ret; asm volatile ("ld.global.nc.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldg(const double *ptr) { double ret; asm volatile ("ld.global.nc.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.nc.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }
# 149 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ long __ldcg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldcg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldcg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldcg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldcg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldcg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldcg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldcg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cg.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldcg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cg.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldcg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cg.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldcg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cg.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldcg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldcg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldcg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cg.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldcg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldcg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldcg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldcg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldcg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cg.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldcg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cg.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldcg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cg.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldcg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cg.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldcg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldcg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldcg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cg.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldcg(const float *ptr) { float ret; asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldcg(const double *ptr) { double ret; asm volatile ("ld.global.cg.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldcg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldcg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldcg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cg.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }







static __attribute__((device)) __inline__ long __ldca(const long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldca(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldca(const char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldca(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldca(const short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldca(const int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldca(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldca(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.ca.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldca(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.ca.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldca(const short2 *ptr) { short2 ret; asm volatile ("ld.global.ca.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldca(const short4 *ptr) { short4 ret; asm volatile ("ld.global.ca.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldca(const int2 *ptr) { int2 ret; asm volatile ("ld.global.ca.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldca(const int4 *ptr) { int4 ret; asm volatile ("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldca(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.ca.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldca(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldca(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldca(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldca(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldca(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.ca.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldca(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.ca.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldca(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.ca.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldca(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.ca.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldca(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldca(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldca(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldca(const float *ptr) { float ret; asm volatile ("ld.global.ca.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldca(const double *ptr) { double ret; asm volatile ("ld.global.ca.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldca(const float2 *ptr) { float2 ret; asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldca(const float4 *ptr) { float4 ret; asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldca(const double2 *ptr) { double2 ret; asm volatile ("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }







static __attribute__((device)) __inline__ long __ldcs(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldcs(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldcs(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldcs(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldcs(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldcs(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldcs(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldcs(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cs.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldcs(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cs.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldcs(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cs.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldcs(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cs.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldcs(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cs.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldcs(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldcs(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cs.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldcs(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldcs(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldcs(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldcs(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldcs(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cs.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldcs(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cs.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldcs(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cs.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldcs(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cs.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldcs(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cs.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldcs(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldcs(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cs.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldcs(const float *ptr) { float ret; asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldcs(const double *ptr) { double ret; asm volatile ("ld.global.cs.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldcs(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldcs(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldcs(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cs.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }
# 286 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
static __attribute__((device)) __inline__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}


static __attribute__((device)) __inline__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
static __attribute__((device)) __inline__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
# 294 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h" 2
# 3271 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_intrinsics.h" 1
# 111 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_32_intrinsics.h" 1
# 112 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_35_intrinsics.h" 2
# 3272 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h" 2
# 91 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h"
static __attribute__((device)) __inline__ int __dp2a_lo(int srcA, int srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c) ;

static __attribute__((device)) __inline__ int __dp2a_lo(short2 srcA, char4 srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c) ;

static __attribute__((device)) __inline__ int __dp2a_hi(int srcA, int srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c) ;

static __attribute__((device)) __inline__ int __dp2a_hi(short2 srcA, char4 srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c) ;






static __attribute__((device)) __inline__ int __dp4a(int srcA, int srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c) ;

static __attribute__((device)) __inline__ int __dp4a(char4 srcA, char4 srcB, int c) ;
static __attribute__((device)) __inline__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c) ;
# 122 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.hpp" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 72 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.hpp" 2
# 81 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.hpp"
static __attribute__((device)) __inline__ int __dp4a(int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ int __dp4a(char4 srcA, char4 srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(int *)&srcA), "r"(*(int *)&srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(unsigned int *)&srcA), "r"(*(unsigned int *)&srcB), "r"(c));
    return ret;
}


static __attribute__((device)) __inline__ int __dp2a_lo(int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ int __dp2a_lo(short2 srcA, char4 srcB, int c) {
    int ret;
    asm volatile ("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(int *)&srcA), "r"(*(int *)&srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(unsigned int *)&srcA), "r"(*(unsigned int *)&srcB), "r"(c));
    return ret;
}


static __attribute__((device)) __inline__ int __dp2a_hi(int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ int __dp2a_hi(short2 srcA, char4 srcB, int c) {
    int ret;
    asm volatile ("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(int *)&srcA), "r"(*(int *)&srcB), "r"(c));
    return ret;
}

static __attribute__((device)) __inline__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(*(unsigned int *)&srcA), "r"(*(unsigned int *)&srcB), "r"(c));
    return ret;
}
# 123 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/sm_61_intrinsics.h" 2
# 3273 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h" 2
# 83 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h"
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, int value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned long value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, long value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned long long value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, long long value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, float value) ;
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, double value) ;

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, int value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned long value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, long value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned long long value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, long long value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, float value, int *pred) ;
static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, double value, int *pred) ;

static __attribute__((device)) __inline__ void __nanosleep(unsigned int ns) ;
# 111 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.hpp" 1
# 69 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.hpp"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 70 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.hpp" 2
# 83 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.hpp"
static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned value) {
  return __match32_any_sync(mask, value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, int value) {
  return __match32_any_sync(mask, value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned long value) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_any_sync(mask, (unsigned long long)value):
    __match32_any_sync(mask, (unsigned)value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, long value) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_any_sync(mask, (unsigned long long)value):
    __match32_any_sync(mask, (unsigned)value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, unsigned long long value) {
  return __match64_any_sync(mask, value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, long long value) {
  return __match64_any_sync(mask, value);
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, float value) {
  return __match32_any_sync(mask, float_as_uint(value));
}

static __attribute__((device)) __inline__ unsigned int __match_any_sync(unsigned mask, double value) {
  return __match64_any_sync(mask, __double_as_longlong(value));
}




static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned value, int *pred) {
  return __match32_all_sync(mask, value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, int value, int *pred) {
  return __match32_all_sync(mask, value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned long value, int *pred) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_all_sync(mask, (unsigned long long)value, pred):
    __match32_all_sync(mask, (unsigned)value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, long value, int *pred) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_all_sync(mask, (unsigned long long)value, pred):
    __match32_all_sync(mask, (unsigned)value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, unsigned long long value, int *pred) {
  return __match64_all_sync(mask, value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, long long value, int *pred) {
  return __match64_all_sync(mask, value, pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, float value, int *pred) {
  return __match32_all_sync(mask, float_as_uint(value), pred);
}

static __attribute__((device)) __inline__ unsigned int __match_all_sync(unsigned mask, double value, int *pred) {
  return __match64_all_sync(mask, __double_as_longlong(value), pred);
}

static __attribute__((device)) __inline__ void __nanosleep(unsigned int ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}
# 112 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/sm_70_rt.h" 2
# 3274 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_functions.h" 1
# 61 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_functions.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 64 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_functions.h" 2



template <typename T> struct __nv_surf_trait { typedef void * cast_type; };

template<> struct __nv_surf_trait<char> { typedef char * cast_type; };
template<> struct __nv_surf_trait<signed char> { typedef signed char * cast_type; };
template<> struct __nv_surf_trait<unsigned char> { typedef unsigned char * cast_type; };
template<> struct __nv_surf_trait<char1> { typedef char1 * cast_type; };
template<> struct __nv_surf_trait<uchar1> { typedef uchar1 * cast_type; };
template<> struct __nv_surf_trait<char2> { typedef char2 * cast_type; };
template<> struct __nv_surf_trait<uchar2> { typedef uchar2 * cast_type; };
template<> struct __nv_surf_trait<char4> { typedef char4 * cast_type; };
template<> struct __nv_surf_trait<uchar4> { typedef uchar4 * cast_type; };
template<> struct __nv_surf_trait<short> { typedef short * cast_type; };
template<> struct __nv_surf_trait<unsigned short> { typedef unsigned short * cast_type; };
template<> struct __nv_surf_trait<short1> { typedef short1 * cast_type; };
template<> struct __nv_surf_trait<ushort1> { typedef ushort1 * cast_type; };
template<> struct __nv_surf_trait<short2> { typedef short2 * cast_type; };
template<> struct __nv_surf_trait<ushort2> { typedef ushort2 * cast_type; };
template<> struct __nv_surf_trait<short4> { typedef short4 * cast_type; };
template<> struct __nv_surf_trait<ushort4> { typedef ushort4 * cast_type; };
template<> struct __nv_surf_trait<int> { typedef int * cast_type; };
template<> struct __nv_surf_trait<unsigned int> { typedef unsigned int * cast_type; };
template<> struct __nv_surf_trait<int1> { typedef int1 * cast_type; };
template<> struct __nv_surf_trait<uint1> { typedef uint1 * cast_type; };
template<> struct __nv_surf_trait<int2> { typedef int2 * cast_type; };
template<> struct __nv_surf_trait<uint2> { typedef uint2 * cast_type; };
template<> struct __nv_surf_trait<int4> { typedef int4 * cast_type; };
template<> struct __nv_surf_trait<uint4> { typedef uint4 * cast_type; };
template<> struct __nv_surf_trait<long long> { typedef long long * cast_type; };
template<> struct __nv_surf_trait<unsigned long long> { typedef unsigned long long * cast_type; };
template<> struct __nv_surf_trait<longlong1> { typedef longlong1 * cast_type; };
template<> struct __nv_surf_trait<ulonglong1> { typedef ulonglong1 * cast_type; };
template<> struct __nv_surf_trait<longlong2> { typedef longlong2 * cast_type; };
template<> struct __nv_surf_trait<ulonglong2> { typedef ulonglong2 * cast_type; };
# 110 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_functions.h"
template<> struct __nv_surf_trait<float> { typedef float * cast_type; };
template<> struct __nv_surf_trait<float1> { typedef float1 * cast_type; };
template<> struct __nv_surf_trait<float2> { typedef float2 * cast_type; };
template<> struct __nv_surf_trait<float4> { typedef float4 * cast_type; };


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dread(T *res, surface<void, 0x01> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dread_v2", (void *)res, s, surf, x, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf1Dread(surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf1Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dread(T *res, surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf1Dread<T>(surf, x, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dread(T *res, surface<void, 0x02> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dread_v2", (void *)res, s, surf, x, y, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf2Dread(surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf2Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dread(T *res, surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf2Dread<T>(surf, x, y, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dread(T *res, surface<void, 0x03> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dread_v2", (void *)res, s, surf, x, y, z, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf3Dread(surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf3Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, z, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dread(T *res, surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf3Dread<T>(surf, x, y, z, mode);

}



template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredread(T *res, surface<void, 0xF1> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredread_v2", (void *)res, s, surf, x, layer, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf1DLayeredread(surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf1DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, layer, mode);
  return temp;

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredread(T *res, surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf1DLayeredread<T>(surf, x, layer, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredread(T *res, surface<void, 0xF2> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredread_v2", (void *)res, s, surf, x, y, layer, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf2DLayeredread(surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf2DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layer, mode);
  return temp;

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredread(T *res, surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf2DLayeredread<T>(surf, x, y, layer, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapread(T *res, surface<void, 0x0C> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapread_v2", (void *)res, s, surf, x, y, face, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surfCubemapread(surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;

  __nv_tex_surf_handler("__surfCubemapread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, face, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapread(T *res, surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surfCubemapread<T>(surf, x, y, face, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredread(T *res, surface<void, 0xFC> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (void *)res, s, surf, x, y, layerFace, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surfCubemapLayeredread(surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layerFace, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredread(T *res, surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surfCubemapLayeredread<T>(surf, x, y, layerFace, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dwrite(T val, surface<void, 0x01> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dwrite_v2", (void *)&val, s, surf, x, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dwrite(T val, surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, mode);

}



template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dwrite(T val, surface<void, 0x02> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dwrite_v2", (void *)&val, s, surf, x, y, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dwrite(T val, surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dwrite(T val, surface<void, 0x03> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dwrite_v2", (void *)&val, s, surf, x, y, z,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dwrite(T val, surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, z, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredwrite(T val, surface<void, 0xF1> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (void *)&val, s, surf, x, layer,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredwrite(T val, surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, layer, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredwrite(T val, surface<void, 0xF2> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (void *)&val, s, surf, x, y, layer,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredwrite(T val, surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, layer, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapwrite(T val, surface<void, 0x0C> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapwrite_v2", (void *)&val, s, surf, x, y, face, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapwrite(T val, surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, face, mode);

}



template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredwrite(T val, surface<void, 0xFC> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (void *)&val, s, surf, x, y, layerFace, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredwrite(T val, surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, layerFace, mode);

}
# 3275 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h" 1
# 62 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 63 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h" 2

# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 65 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h" 2



template <typename T>
struct __nv_tex_rmet_ret { };

template<> struct __nv_tex_rmet_ret<char> { typedef char type; };
template<> struct __nv_tex_rmet_ret<signed char> { typedef signed char type; };
template<> struct __nv_tex_rmet_ret<unsigned char> { typedef unsigned char type; };
template<> struct __nv_tex_rmet_ret<char1> { typedef char1 type; };
template<> struct __nv_tex_rmet_ret<uchar1> { typedef uchar1 type; };
template<> struct __nv_tex_rmet_ret<char2> { typedef char2 type; };
template<> struct __nv_tex_rmet_ret<uchar2> { typedef uchar2 type; };
template<> struct __nv_tex_rmet_ret<char4> { typedef char4 type; };
template<> struct __nv_tex_rmet_ret<uchar4> { typedef uchar4 type; };

template<> struct __nv_tex_rmet_ret<short> { typedef short type; };
template<> struct __nv_tex_rmet_ret<unsigned short> { typedef unsigned short type; };
template<> struct __nv_tex_rmet_ret<short1> { typedef short1 type; };
template<> struct __nv_tex_rmet_ret<ushort1> { typedef ushort1 type; };
template<> struct __nv_tex_rmet_ret<short2> { typedef short2 type; };
template<> struct __nv_tex_rmet_ret<ushort2> { typedef ushort2 type; };
template<> struct __nv_tex_rmet_ret<short4> { typedef short4 type; };
template<> struct __nv_tex_rmet_ret<ushort4> { typedef ushort4 type; };

template<> struct __nv_tex_rmet_ret<int> { typedef int type; };
template<> struct __nv_tex_rmet_ret<unsigned int> { typedef unsigned int type; };
template<> struct __nv_tex_rmet_ret<int1> { typedef int1 type; };
template<> struct __nv_tex_rmet_ret<uint1> { typedef uint1 type; };
template<> struct __nv_tex_rmet_ret<int2> { typedef int2 type; };
template<> struct __nv_tex_rmet_ret<uint2> { typedef uint2 type; };
template<> struct __nv_tex_rmet_ret<int4> { typedef int4 type; };
template<> struct __nv_tex_rmet_ret<uint4> { typedef uint4 type; };
# 109 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret<float> { typedef float type; };
template<> struct __nv_tex_rmet_ret<float1> { typedef float1 type; };
template<> struct __nv_tex_rmet_ret<float2> { typedef float2 type; };
template<> struct __nv_tex_rmet_ret<float4> { typedef float4 type; };


template <typename T> struct __nv_tex_rmet_cast { typedef T* type; };
# 127 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_fetch_functions.h"
template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1Dfetch(texture<T, 0x01, cudaReadModeElementType> t, int x)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1Dfetch_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x);
  return temp;

}

template <typename T>
struct __nv_tex_rmnf_ret { };

template <> struct __nv_tex_rmnf_ret<char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<signed char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<char1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<uchar1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<short1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<ushort1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<char2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<uchar2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<short2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<ushort2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<char4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<uchar4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<short4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1Dfetch(texture<T, 0x01, cudaReadModeNormalizedFloat> t, int x)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1Dfetch_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1D(texture<T, 0x01, cudaReadModeElementType> t, float x)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1D(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1D_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2D(texture<T, 0x02, cudaReadModeElementType> t, float x, float y)
{

  typename __nv_tex_rmet_ret<T>::type temp;

  __nv_tex_surf_handler("__tex2D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2D(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2D_rmnf_v2", &type_dummy, &retval, t, x, y);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayered(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayered_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayered(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayered_rmnf_v2", &type_dummy, &retval, t, x, layer);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayered(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayered_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayered(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayered_rmnf_v2", &type_dummy, &retval, t, x, y, layer);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3D(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3D(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3D_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemap(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemap_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemap(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemap_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;

}


template <typename T>
struct __nv_tex2dgather_ret { };
template <> struct __nv_tex2dgather_ret<char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<signed char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char1> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char2> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char3> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char4> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<unsigned char> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar1> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar2> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar3> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar4> { typedef uchar4 type; };

template <> struct __nv_tex2dgather_ret<short> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short1> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short2> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short3> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short4> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<unsigned short> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort1> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort2> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort3> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort4> { typedef ushort4 type; };

template <> struct __nv_tex2dgather_ret<int> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int1> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int2> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int3> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int4> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<unsigned int> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint1> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint2> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint3> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint4> { typedef uint4 type; };

template <> struct __nv_tex2dgather_ret<float> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float1> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float2> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float3> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float4> { typedef float4 type; };

template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) typename __nv_tex2dgather_ret<T>::type tex2Dgather(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, int comp=0)
{

  T type_dummy;
  typename __nv_tex2dgather_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2Dgather_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;

}


template<typename T> struct __nv_tex2dgather_rmnf_ret { };
template<> struct __nv_tex2dgather_rmnf_ret<char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) typename __nv_tex2dgather_rmnf_ret<T>::type tex2Dgather(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0)
{

  T type_dummy;
  typename __nv_tex2dgather_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2Dgather_rmnf_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLod(texture<T, 0x01, cudaReadModeElementType> t, float x, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLod(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLod_rmnf_v2", &type_dummy, &retval, t, x, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLod(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLod(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLod_rmnf_v2", &type_dummy, &retval, t, x, y, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayeredLod(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayeredLod(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, layer, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayeredLod(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayeredLod(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, layer, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3DLod(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3DLod_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3DLod(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3DLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLod(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLod_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLod(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayered(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayered_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayered(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayered_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayeredLod(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayeredLod(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer, level);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapGrad(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapGrad(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapGrad_rmnf_v2", &type_dummy, &retval, t, x, y, z, &dPdx, &dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayeredGrad(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayeredGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayeredGrad(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, layer, &dPdx, &dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DGrad(texture<T, 0x01, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, dPdx, dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DGrad(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DGrad_rmnf_v2", &type_dummy, &retval,t, x,dPdx, dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DGrad(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DGrad(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, &dPdx, &dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayeredGrad(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayeredGrad_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, layer, dPdx, dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayeredGrad(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, layer, dPdx, dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayeredGrad(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayeredGrad_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, layer, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayeredGrad(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, layer, &dPdx, &dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3DGrad(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3DGrad(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, &dPdx, &dPdy);
  return retval;

}
# 3276 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_indirect_functions.h" 1
# 57 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_indirect_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 58 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_indirect_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 59 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_indirect_functions.h" 2


template <typename T> struct __nv_itex_trait { };
template<> struct __nv_itex_trait<char> { typedef void type; };
template<> struct __nv_itex_trait<signed char> { typedef void type; };
template<> struct __nv_itex_trait<char1> { typedef void type; };
template<> struct __nv_itex_trait<char2> { typedef void type; };
template<> struct __nv_itex_trait<char4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned char> { typedef void type; };
template<> struct __nv_itex_trait<uchar1> { typedef void type; };
template<> struct __nv_itex_trait<uchar2> { typedef void type; };
template<> struct __nv_itex_trait<uchar4> { typedef void type; };
template<> struct __nv_itex_trait<short> { typedef void type; };
template<> struct __nv_itex_trait<short1> { typedef void type; };
template<> struct __nv_itex_trait<short2> { typedef void type; };
template<> struct __nv_itex_trait<short4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned short> { typedef void type; };
template<> struct __nv_itex_trait<ushort1> { typedef void type; };
template<> struct __nv_itex_trait<ushort2> { typedef void type; };
template<> struct __nv_itex_trait<ushort4> { typedef void type; };
template<> struct __nv_itex_trait<int> { typedef void type; };
template<> struct __nv_itex_trait<int1> { typedef void type; };
template<> struct __nv_itex_trait<int2> { typedef void type; };
template<> struct __nv_itex_trait<int4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned int> { typedef void type; };
template<> struct __nv_itex_trait<uint1> { typedef void type; };
template<> struct __nv_itex_trait<uint2> { typedef void type; };
template<> struct __nv_itex_trait<uint4> { typedef void type; };
# 97 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait<float> { typedef void type; };
template<> struct __nv_itex_trait<float1> { typedef void type; };
template<> struct __nv_itex_trait<float2> { typedef void type; };
template<> struct __nv_itex_trait<float4> { typedef void type; };



template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x)
{

   __nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x);

}

template <class T>
static __attribute__((device)) T tex1Dfetch(cudaTextureObject_t texObject, int x)
{

  T ret;
  tex1Dfetch(&ret, texObject, x);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1D(T *ptr, cudaTextureObject_t obj, float x)
{

   __nv_tex_surf_handler("__itex1D", ptr, obj, x);

}


template <class T>
static __attribute__((device)) T tex1D(cudaTextureObject_t texObject, float x)
{

  T ret;
  tex1D(&ret, texObject, x);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2D(T *ptr, cudaTextureObject_t obj, float x, float y)
{

   __nv_tex_surf_handler("__itex2D", ptr, obj, x, y);

}

template <class T>
static __attribute__((device)) T tex2D(cudaTextureObject_t texObject, float x, float y)
{

  T ret;
  tex2D(&ret, texObject, x, y);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{

   __nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z);

}

template <class T>
static __attribute__((device)) T tex3D(cudaTextureObject_t texObject, float x, float y, float z)
{

  T ret;
  tex3D(&ret, texObject, x, y, z);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer)
{

   __nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer);

}

template <class T>
static __attribute__((device)) T tex1DLayered(cudaTextureObject_t texObject, float x, int layer)
{

  T ret;
  tex1DLayered(&ret, texObject, x, layer);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer)
{

  __nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer);

}

template <class T>
static __attribute__((device)) T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer)
{

  T ret;
  tex2DLayered(&ret, texObject, x, y, layer);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{

  __nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z);

}


template <class T>
static __attribute__((device)) T texCubemap(cudaTextureObject_t texObject, float x, float y, float z)
{

  T ret;
  texCubemap(&ret, texObject, x, y, z);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer)
{

  __nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer);

}

template <class T>
static __attribute__((device)) T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer)
{

  T ret;
  texCubemapLayered(&ret, texObject, x, y, z, layer);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0)
{

  __nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp);

}

template <class T>
static __attribute__((device)) T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0)
{

  T ret;
  tex2Dgather(&ret, to, x, y, comp);
  return ret;

}



template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level)
{

  __nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level);

}

template <class T>
static __attribute__((device)) T tex1DLod(cudaTextureObject_t texObject, float x, float level)
{

  T ret;
  tex1DLod(&ret, texObject, x, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level)
{

  __nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level);

}

template <class T>
static __attribute__((device)) T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level)
{

  T ret;
  tex2DLod(&ret, texObject, x, y, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{

  __nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level);

}

template <class T>
static __attribute__((device)) T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{

  T ret;
  tex3DLod(&ret, texObject, x, y, z, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level)
{

  __nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level);

}

template <class T>
static __attribute__((device)) T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level)
{

  T ret;
  tex1DLayeredLod(&ret, texObject, x, layer, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level)
{

  __nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level);

}

template <class T>
static __attribute__((device)) T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level)
{

  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{

  __nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level);

}

template <class T>
static __attribute__((device)) T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{

  T ret;
  texCubemapLod(&ret, texObject, x, y, z, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T ret;
  texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level)
{

  __nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level);

}

template <class T>
static __attribute__((device)) T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{

  T ret;
  texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy)
{

  __nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy);

}

template <class T>
static __attribute__((device)) T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy)
{

  T ret;
  tex1DGrad(&ret, texObject, x, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy)
{

  __nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy);


}

template <class T>
static __attribute__((device)) T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{

  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy)
{

  __nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy);

}

template <class T>
static __attribute__((device)) T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{

  T ret;
  tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayeredGrad(T * ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  __nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  T ret;
  texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy);
  return ret;

}
# 3277 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_indirect_functions.h" 1
# 57 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_indirect_functions.h"
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/builtin_types.h" 1
# 58 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_indirect_functions.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/host_defines.h" 1
# 59 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/surface_indirect_functions.h" 2

template<typename T> struct __nv_isurf_trait { };
template<> struct __nv_isurf_trait<char> { typedef void type; };
template<> struct __nv_isurf_trait<signed char> { typedef void type; };
template<> struct __nv_isurf_trait<char1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned char> { typedef void type; };
template<> struct __nv_isurf_trait<uchar1> { typedef void type; };
template<> struct __nv_isurf_trait<short> { typedef void type; };
template<> struct __nv_isurf_trait<short1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned short> { typedef void type; };
template<> struct __nv_isurf_trait<ushort1> { typedef void type; };
template<> struct __nv_isurf_trait<int> { typedef void type; };
template<> struct __nv_isurf_trait<int1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned int> { typedef void type; };
template<> struct __nv_isurf_trait<uint1> { typedef void type; };
template<> struct __nv_isurf_trait<long long> { typedef void type; };
template<> struct __nv_isurf_trait<longlong1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned long long> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong1> { typedef void type; };
template<> struct __nv_isurf_trait<float> { typedef void type; };
template<> struct __nv_isurf_trait<float1> { typedef void type; };

template<> struct __nv_isurf_trait<char2> { typedef void type; };
template<> struct __nv_isurf_trait<uchar2> { typedef void type; };
template<> struct __nv_isurf_trait<short2> { typedef void type; };
template<> struct __nv_isurf_trait<ushort2> { typedef void type; };
template<> struct __nv_isurf_trait<int2> { typedef void type; };
template<> struct __nv_isurf_trait<uint2> { typedef void type; };
template<> struct __nv_isurf_trait<longlong2> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong2> { typedef void type; };
template<> struct __nv_isurf_trait<float2> { typedef void type; };

template<> struct __nv_isurf_trait<char4> { typedef void type; };
template<> struct __nv_isurf_trait<uchar4> { typedef void type; };
template<> struct __nv_isurf_trait<short4> { typedef void type; };
template<> struct __nv_isurf_trait<ushort4> { typedef void type; };
template<> struct __nv_isurf_trait<int4> { typedef void type; };
template<> struct __nv_isurf_trait<uint4> { typedef void type; };
template<> struct __nv_isurf_trait<float4> { typedef void type; };


template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode);

}

template <class T>
static __attribute__((device)) T surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf1Dread(&ret, surfObject, x, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode);

}

template <class T>
static __attribute__((device)) T surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf2Dread(&ret, surfObject, x, y, boundaryMode);
   return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode);

}

template <class T>
static __attribute__((device)) T surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf3Dread(&ret, surfObject, x, y, z, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode);

}

template <class T>
static __attribute__((device)) T surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode);

}

template <class T>
static __attribute__((device)) T surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode);

}

template <class T>
static __attribute__((device)) T surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surfCubemapread(&ret, surfObject, x, y, face, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode);

}

template <class T>
static __attribute__((device)) T surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode);

}
# 3278 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/crt/device_functions.h" 2


extern "C" __attribute__((host)) __attribute__((device)) unsigned __cudaPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem = 0,
                                      void *stream = 0);
# 50 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_functions.h" 2
# 119 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 1 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_launch_parameters.h" 1
# 68 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/device_launch_parameters.h"
extern "C" {


uint3 __attribute__((device_builtin)) extern const threadIdx;
uint3 __attribute__((device_builtin)) extern const blockIdx;
dim3 __attribute__((device_builtin)) extern const blockDim;
dim3 __attribute__((device_builtin)) extern const gridDim;
int __attribute__((device_builtin)) extern const warpSize;




}
# 120 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h" 2
# 185 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaLaunchKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}
# 245 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaLaunchCooperativeKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}
# 283 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaSetupArgument(
  T arg,
  size_t offset
)
{
  return ::cudaSetupArgument((const void*)&arg, sizeof(T), offset);
}
# 322 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
static __inline__ __attribute__((host)) cudaError_t cudaEventCreate(
  cudaEvent_t *event,
  unsigned int flags
)
{
  return ::cudaEventCreateWithFlags(event, flags);
}
# 385 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
static __inline__ __attribute__((host)) cudaError_t cudaMallocHost(
  void **ptr,
  size_t size,
  unsigned int flags
)
{
  return ::cudaHostAlloc(ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaHostAlloc(
  T **ptr,
  size_t size,
  unsigned int flags
)
{
  return ::cudaHostAlloc((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaHostGetDevicePointer(
  T **pDevice,
  void *pHost,
  unsigned int flags
)
{
  return ::cudaHostGetDevicePointer((void**)(void*)pDevice, pHost, flags);
}
# 512 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocManaged(
  T **devPtr,
  size_t size,
  unsigned int flags = 0x01
)
{
  return ::cudaMallocManaged((void**)(void*)devPtr, size, flags);
}
# 600 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaStreamAttachMemAsync(
  cudaStream_t stream,
  T *devPtr,
  size_t length = 0,
  unsigned int flags = 0x04
)
{
  return ::cudaStreamAttachMemAsync(stream, (void*)devPtr, length, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMalloc(
  T **devPtr,
  size_t size
)
{
  return ::cudaMalloc((void**)(void*)devPtr, size);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocHost(
  T **ptr,
  size_t size,
  unsigned int flags = 0
)
{
  return cudaMallocHost((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocPitch(
  T **devPtr,
  size_t *pitch,
  size_t width,
  size_t height
)
{
  return ::cudaMallocPitch((void**)(void*)devPtr, pitch, width, height);
}
# 676 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyToSymbol(
  const T &symbol,
  const void *src,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice
)
{
  return ::cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}
# 728 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyToSymbolAsync(
  const T &symbol,
  const void *src,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice,
        cudaStream_t stream = 0
)
{
  return ::cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}
# 774 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyFromSymbol(
        void *dst,
  const T &symbol,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost
)
{
  return ::cudaMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}
# 826 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyFromSymbolAsync(
        void *dst,
  const T &symbol,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
        cudaStream_t stream = 0
)
{
  return ::cudaMemcpyFromSymbolAsync(dst, (const void*)&symbol, count, offset, kind, stream);
}
# 860 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGetSymbolAddress(
        void **devPtr,
  const T &symbol
)
{
  return ::cudaGetSymbolAddress(devPtr, (const void*)&symbol);
}
# 890 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGetSymbolSize(
        size_t *size,
  const T &symbol
)
{
  return ::cudaGetSymbolSize(size, (const void*)&symbol);
}
# 932 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTexture(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct cudaChannelFormatDesc &desc,
        size_t size = (2147483647 * 2U + 1U)
)
{
  return ::cudaBindTexture(offset, &tex, devPtr, &desc, size);
}
# 976 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTexture(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
        size_t size = (2147483647 * 2U + 1U)
)
{
  return cudaBindTexture(offset, tex, devPtr, tex.channelDesc, size);
}
# 1031 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTexture2D(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct cudaChannelFormatDesc &desc,
  size_t width,
  size_t height,
  size_t pitch
)
{
  return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}
# 1088 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTexture2D(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  size_t width,
  size_t height,
  size_t pitch
)
{
  return ::cudaBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
# 1129 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  cudaArray_const_t array,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindTextureToArray(&tex, array, &desc);
}
# 1166 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  cudaArray_const_t array
)
{
  struct cudaChannelFormatDesc desc;
  cudaError_t err = ::cudaGetChannelDesc(&desc, array);

  return err == cudaSuccess ? cudaBindTextureToArray(tex, array, desc) : err;
}
# 1206 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(
  const struct texture<T, dim, readMode> &tex,
  cudaMipmappedArray_const_t mipmappedArray,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc);
}
# 1243 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(
  const struct texture<T, dim, readMode> &tex,
  cudaMipmappedArray_const_t mipmappedArray
)
{
  struct cudaChannelFormatDesc desc;
  cudaArray_t levelArray;
  cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0);

  if (err != cudaSuccess) {
      return err;
  }
  err = ::cudaGetChannelDesc(&desc, levelArray);

  return err == cudaSuccess ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err;
}
# 1284 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaUnbindTexture(
  const struct texture<T, dim, readMode> &tex
)
{
  return ::cudaUnbindTexture(&tex);
}
# 1318 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __inline__ __attribute__((host)) cudaError_t cudaGetTextureAlignmentOffset(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex
)
{
  return ::cudaGetTextureAlignmentOffset(offset, &tex);
}
# 1370 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetCacheConfig(
  T *func,
  enum cudaFuncCache cacheConfig
)
{
  return ::cudaFuncSetCacheConfig((const void*)func, cacheConfig);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetSharedMemConfig(
  T *func,
  enum cudaSharedMemConfig config
)
{
  return ::cudaFuncSetSharedMemConfig((const void*)func, config);
}
# 1415 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks,
    T func,
    int blockSize,
    size_t dynamicSMemSize)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, 0x00);
}
# 1466 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks,
    T func,
    int blockSize,
    size_t dynamicSMemSize,
    unsigned int flags)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, flags);
}




class __cudaOccupancyB2DHelper {
  size_t n;
public:
  inline __attribute__((host)) __attribute__((device)) __cudaOccupancyB2DHelper(size_t n_) : n(n_) {}
  inline __attribute__((host)) __attribute__((device)) size_t operator()(int)
  {
      return n;
  }
};
# 1535 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<typename UnaryFunction, class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
    int *minGridSize,
    int *blockSize,
    T func,
    UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0,
    unsigned int flags = 0)
{
    cudaError_t status;


    int device;
    struct cudaFuncAttributes attr;


    int maxThreadsPerMultiProcessor;
    int warpSize;
    int devMaxThreadsPerBlock;
    int multiProcessorCount;
    int funcMaxThreadsPerBlock;
    int occupancyLimit;
    int granularity;


    int maxBlockSize = 0;
    int numBlocks = 0;
    int maxOccupancy = 0;


    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int occupancyInBlocks;
    int occupancyInThreads;
    size_t dynamicSMemSize;





    if (!minGridSize || !blockSize || !func) {
        return cudaErrorInvalidValue;
    }





    status = ::cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &maxThreadsPerMultiProcessor,
        cudaDevAttrMaxThreadsPerMultiProcessor,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &warpSize,
        cudaDevAttrWarpSize,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &devMaxThreadsPerBlock,
        cudaDevAttrMaxThreadsPerBlock,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &multiProcessorCount,
        cudaDevAttrMultiProcessorCount,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaFuncGetAttributes(&attr, func);
    if (status != cudaSuccess) {
        return status;
    }

    funcMaxThreadsPerBlock = attr.maxThreadsPerBlock;





    occupancyLimit = maxThreadsPerMultiProcessor;
    granularity = warpSize;

    if (blockSizeLimit == 0) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (devMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (funcMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = funcMaxThreadsPerBlock;
    }

    blockSizeLimitAligned = ((blockSizeLimit + (granularity - 1)) / granularity) * granularity;

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {



        if (blockSizeLimit < blockSizeToTryAligned) {
            blockSizeToTry = blockSizeLimit;
        } else {
            blockSizeToTry = blockSizeToTryAligned;
        }

        dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

        status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &occupancyInBlocks,
            func,
            blockSizeToTry,
            dynamicSMemSize,
            flags);

        if (status != cudaSuccess) {
            return status;
        }

        occupancyInThreads = blockSizeToTry * occupancyInBlocks;

        if (occupancyInThreads > maxOccupancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks = occupancyInBlocks;
            maxOccupancy = occupancyInThreads;
        }



        if (occupancyLimit == maxOccupancy) {
            break;
        }
    }







    *minGridSize = numBlocks * multiProcessorCount;
    *blockSize = maxBlockSize;

    return status;
}
# 1730 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<typename UnaryFunction, class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    int *minGridSize,
    int *blockSize,
    T func,
    UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0x00);
}
# 1775 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int *minGridSize,
    int *blockSize,
    T func,
    size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0)
{
  return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, 0x00);
}
# 1834 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize,
    int *blockSize,
    T func,
    size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0,
    unsigned int flags = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, flags);
}
# 1885 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaLaunch(
  T *func
)
{
  return ::cudaLaunch((const void*)func);
}
# 1922 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncGetAttributes(
  struct cudaFuncAttributes *attr,
  T *entry
)
{
  return ::cudaFuncGetAttributes(attr, (const void*)entry);
}
# 1967 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetAttribute(
  T *entry,
  enum cudaFuncAttribute attr,
  int value
)
{
  return ::cudaFuncSetAttribute((const void*)entry, attr, value);
}
# 1997 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim>
static __inline__ __attribute__((host)) cudaError_t cudaBindSurfaceToArray(
  const struct surface<T, dim> &surf,
  cudaArray_const_t array,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindSurfaceToArray(&surf, array, &desc);
}
# 2026 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
template<class T, int dim>
static __inline__ __attribute__((host)) cudaError_t cudaBindSurfaceToArray(
  const struct surface<T, dim> &surf,
  cudaArray_const_t array
)
{
  struct cudaChannelFormatDesc desc;
  cudaError_t err = ::cudaGetChannelDesc(&desc, array);

  return err == cudaSuccess ? cudaBindSurfaceToArray(surf, array, desc) : err;
}
# 2047 "/usr/tce/packages/cuda/cuda-9.2.148/nvidia/bin/../targets/ppc64le-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 1 "<command-line>" 2
# 1 "comp.cu"

# 1 "comp.h" 1


__attribute__((device)) void comp();
__attribute__((host)) void comp_host();
# 3 "comp.cu" 2

__attribute__((device)) double calc() {

  return 3.4*4.5;

}
