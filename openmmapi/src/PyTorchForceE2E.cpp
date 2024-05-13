#include "PyTorchForce.h"
#include "internal/PyTorchForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <vector>
using std::vector;

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2E::PyTorchForceE2E(const std::string& file,
								 const std::vector<int> particleIndices,
								 const std::vector<double> signalForceWeights,
								 const double scale):

  file(file),
  particleIndices(particleIndices),
  signalForceWeights(signalForceWeights),
  scale(scale),
  usePeriodic(false),
  {
}

const string& PyTorchForceE2E::getFile() const {
  return file;
}
const double PyTorchForceE2E::getScale() const {
  return scale;
}
const std::vector<int> PyTorchForceE2E::getParticleIndices() const{
  return particleIndices;
}
const std::vector<double> PyTorchForceE2E::getSignalForceWeights() const{
  return signalForceWeights;
}
ForceImpl* PyTorchForceE2E::createImpl() const {
  return new PyTorchForceE2EImpl(*this);
}

void PyTorchForceE2E::setUsesPeriodicBoundaryConditions(bool periodic) {
	usePeriodic = periodic;
}

bool PyTorchForceE2E::usesPeriodicBoundaryConditions() const {
	return usePeriodic;
}


int PyTorchForce::addGlobalParameter(const string& name, double defaultValue) {
	globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
	return globalParameters.size()-1;
}

int PyTorchForce::getNumGlobalParameters() const {
	return globalParameters.size();
}

const string& PyTorchForce::getGlobalParameterName(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].name;
}

void PyTorchForce::setGlobalParameterName(int index, const string& name) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].name = name;
}

double PyTorchForce::getGlobalParameterDefaultValue(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].defaultValue;
}

void PyTorchForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].defaultValue = defaultValue;
}
