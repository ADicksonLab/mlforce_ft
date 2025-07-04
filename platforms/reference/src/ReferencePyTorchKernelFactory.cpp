#include "ReferencePyTorchKernelFactory.h"
#include "ReferencePyTorchKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    int argc = 0;
    vector<char**> argv = {NULL};
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
	  Platform& platform = Platform::getPlatform(i);
	  if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
	    ReferencePyTorchKernelFactory* factory = new ReferencePyTorchKernelFactory();
	    platform.registerKernelFactory(CalcPyTorchForceKernel::Name(), factory);
		platform.registerKernelFactory(CalcPyTorchForceE2EKernel::Name(), factory);
		platform.registerKernelFactory(CalcPyTorchForceE2EDirectKernel::Name(), factory);
		platform.registerKernelFactory(CalcPyTorchForceE2EDiffConfKernel::Name(), factory);
	  }
    }
}

extern "C" OPENMM_EXPORT void registerPyTorchReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferencePyTorchKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcPyTorchForceKernel::Name())
	return new ReferenceCalcPyTorchForceKernel(name, platform);
	if (name == CalcPyTorchForceE2EKernel::Name())
	return new ReferenceCalcPyTorchForceE2EKernel(name, platform);
	if (name == CalcPyTorchForceE2EDirectKernel::Name())
	return new ReferenceCalcPyTorchForceE2EDirectKernel(name, platform);
	if (name == CalcPyTorchForceE2EDiffConfKernel::Name())
	return new ReferenceCalcPyTorchForceE2EDiffConfKernel(name, platform);

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
