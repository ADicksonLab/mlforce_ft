#include "internal/PyTorchForceImpl.h"
#include "PyTorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EDiffConfImpl::PyTorchForceE2EDiffConfImpl(const PyTorchForceE2EDiffConf& owner) : owner(owner){
}

PyTorchForceE2EDiffConfImpl::~PyTorchForceE2EDiffConfImpl() {

}

void PyTorchForceE2EDiffConfImpl::initialize(ContextImpl& context) {


  // Gets the model name and opens it.
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    nnModule = torch::jit::load(owner.getFile());
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading model:" << owner.getFile() << "\n";
  }
  kernel = context.getPlatform().createKernel(CalcPyTorchForceE2EDiffConfKernel::Name(), context);
  kernel.getAs<CalcPyTorchForceE2EDiffConfKernel>().initialize(context.getSystem(), owner, nnModule);

}

double PyTorchForceE2EDiffConfImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
	if ((groups&(1<<owner.getForceGroup())) != 0)
	  return kernel.getAs<CalcPyTorchForceE2EDiffConfKernel>().execute(context, includeForces, includeEnergy);
	return 0.0;
}

std::vector<std::string> PyTorchForceE2EDiffConfImpl::getKernelNames() {
	std::vector<std::string> names;
	names.push_back(CalcPyTorchForceE2EDiffConfKernel::Name());
	return names;
}
