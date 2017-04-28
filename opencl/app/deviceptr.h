#ifndef __DEVICEPTR_H__
#define __DEVICEPTR_H__

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif



typedef cl_mem  device_ptr_t;

// TODO: helper functions



#ifdef __cplusplus
}
#endif


#endif

