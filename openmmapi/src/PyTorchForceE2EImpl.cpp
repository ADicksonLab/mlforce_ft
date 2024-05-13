#include "internal/PyTorchForceImpl.h"
#include "PyTorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EImpl::PyTorchForceE2EImpl(const PyTorchForceE2E& owner) : owner(owner){
}

PyTorchForceE2EImpl::~PyTorchForceE2EImpl() {

}

void PyTorchForceE2EImpl::initialize(ContextImpl& context) {


  // Gets the model name and opens it.
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    nnModule = torch::jit::load(owner.getFile());
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  kernel = context.getPlatform().createKernel(CalcPyTorchForceE2EKernel::Name(), context);
  kernel.getAs<CalcPyTorchForceE2EKernel>().initialize(context.getSystem(), owner, nnModule);

}

double PyTorchForceE2EImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
	if ((groups&(1<<owner.getForceGroup())) != 0)
	  return kernel.getAs<CalcPyTorchForceE2EKernel>().execute(context, includeForces, includeEnergy);
	return 0.0;
}

std::vector<std::string> PyTorchForceE2EImpl::getKernelNames() {
	std::vector<std::string> names;
	names.push_back(CalcPyTorchForceE2EKernel::Name());
	return names;
}
