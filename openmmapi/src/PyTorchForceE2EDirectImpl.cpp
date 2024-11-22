#include "internal/PyTorchForceImpl.h"
#include "PyTorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EDirectImpl::PyTorchForceE2EDirectImpl(const PyTorchForceE2EDirect& owner) : owner(owner){
}

PyTorchForceE2EDirectImpl::~PyTorchForceE2EDirectImpl() {

}

void PyTorchForceE2EDirectImpl::initialize(ContextImpl& context) {


  // Gets the model name and opens it.
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    nnModule = torch::jit::load(owner.getFile());
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  kernel = context.getPlatform().createKernel(CalcPyTorchForceE2EDirectKernel::Name(), context);
  kernel.getAs<CalcPyTorchForceE2EDirectKernel>().initialize(context.getSystem(), owner, nnModule);

}

double PyTorchForceE2EDirectImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
	if ((groups&(1<<owner.getForceGroup())) != 0)
	  return kernel.getAs<CalcPyTorchForceE2EDirectKernel>().execute(context, includeForces, includeEnergy);
	return 0.0;
}

std::vector<std::string> PyTorchForceE2EDirectImpl::getKernelNames() {
	std::vector<std::string> names;
	names.push_back(CalcPyTorchForceE2EDirectKernel::Name());
	return names;
}
