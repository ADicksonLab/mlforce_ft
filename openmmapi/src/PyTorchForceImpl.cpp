#include "internal/PyTorchForceImpl.h"
#include "PyTorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceImpl::PyTorchForceImpl(const PyTorchForce& owner) : owner(owner){
}

PyTorchForceImpl::~PyTorchForceImpl() {

}

void PyTorchForceImpl::initialize(ContextImpl& context) {


  // Gets the model name and opens it.
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    nnModule = torch::jit::load(owner.getFile());
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  kernel = context.getPlatform().createKernel(CalcPyTorchForceKernel::Name(), context);
  kernel.getAs<CalcPyTorchForceKernel>().initialize(context.getSystem(), owner, nnModule);

}

double PyTorchForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
	if ((groups&(1<<owner.getForceGroup())) != 0)
	  return kernel.getAs<CalcPyTorchForceKernel>().execute(context, includeForces, includeEnergy);
	return 0.0;
}

std::vector<std::string> PyTorchForceImpl::getKernelNames() {
	std::vector<std::string> names;
	names.push_back(CalcPyTorchForceKernel::Name());
	return names;
}
