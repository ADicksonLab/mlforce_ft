#include "PyTorchForce.h"
#include "internal/PyTorchForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <vector>
using std::vector;

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EDirect::PyTorchForceE2EDirect(const std::string& file,
											 const std::vector<int> particleIndices,
											 const std::vector<double> signalForceWeights,
											 const double scale,
											 const std::vector<torch::Tensor> fixedInputs,
											 const bool useAttr):

  file(file),
  particleIndices(particleIndices),
  signalForceWeights(signalForceWeights),
  scale(scale),
  fixedInputs(fixedInputs),
  useAttr(useAttr),
  usePeriodic(false)
  {
}

const string& PyTorchForceE2EDirect::getFile() const {
  return file;
}
const double PyTorchForceE2EDirect::getScale() const {
  return scale;
}
const std::vector<torch::Tensor> PyTorchForceE2EDirect::getFixedInputs() const {
  return fixedInputs;
}

const bool PyTorchForceE2EDirect::getUseAttr() const {
  return useAttr;
}

const std::vector<int> PyTorchForceE2EDirect::getParticleIndices() const{
  return particleIndices;
}
const std::vector<double> PyTorchForceE2EDirect::getSignalForceWeights() const{
  return signalForceWeights;
}
ForceImpl* PyTorchForceE2EDirect::createImpl() const {
  return new PyTorchForceE2EDirectImpl(*this);
}

void PyTorchForceE2EDirect::setUsesPeriodicBoundaryConditions(bool periodic) {
	usePeriodic = periodic;
}

bool PyTorchForceE2EDirect::usesPeriodicBoundaryConditions() const {
	return usePeriodic;
}

int PyTorchForceE2EDirect::addGlobalParameter(const string& name, double defaultValue) {
	globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
	return globalParameters.size()-1;
}

int PyTorchForceE2EDirect::getNumGlobalParameters() const {
	return globalParameters.size();
}

const string& PyTorchForceE2EDirect::getGlobalParameterName(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].name;
}

void PyTorchForceE2EDirect::setGlobalParameterName(int index, const string& name) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].name = name;
}

double PyTorchForceE2EDirect::getGlobalParameterDefaultValue(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].defaultValue;
}

void PyTorchForceE2EDirect::setGlobalParameterDefaultValue(int index, double defaultValue) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].defaultValue = defaultValue;
}
