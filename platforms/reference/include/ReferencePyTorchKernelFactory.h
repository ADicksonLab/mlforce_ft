#ifndef OPENMM_REFERENCE_Py_Torch_KERNEL_FACTORY_H_
#define OPENMM_REFERENCE_Py_Torch_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the neural network plugin.
 */

class ReferencePyTorchKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};
  
} // namespace OpenMM

#endif /*OPENMM_REFERENCE_Py_Torch_KERNEL_FACTORY_H_*/
