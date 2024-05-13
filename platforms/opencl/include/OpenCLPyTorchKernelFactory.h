#ifndef OPENMM_OPENCL_PY_TORCH_KERNEL_FACTORY_H_
#define OPENMM_OPENCL_PY_TORCH_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace PyTorchPlugin {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the torchml plugin.
 */

class OpenCLPyTorchKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace NNPlugin

#endif /*OPENMM_OPENCL_PY_TORCH_KERNEL_FACTORY_H_*/
