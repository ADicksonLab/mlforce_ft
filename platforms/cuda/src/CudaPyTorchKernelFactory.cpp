#include <exception>

#include "CudaPyTorchKernelFactory.h"
#include "CudaPyTorchKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaPyTorchKernelFactory* factory = new CudaPyTorchKernelFactory();
        platform.registerKernelFactory(CalcPyTorchForceKernel::Name(), factory);
		platform.registerKernelFactory(CalcPyTorchForceE2EKernel::Name(), factory);
		platform.registerKernelFactory(CalcPyTorchForceE2EDirectKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerPyTorchCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaPyTorchKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcPyTorchForceKernel::Name())
        return new CudaCalcPyTorchForceKernel(name, platform, cu);
    if (name == CalcPyTorchForceE2EKernel::Name())
        return new CudaCalcPyTorchForceE2EKernel(name, platform, cu);
    if (name == CalcPyTorchForceE2EDirectKernel::Name())
        return new CudaCalcPyTorchForceE2EDirectKernel(name, platform, cu);

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
