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
								 const double scale,
								 const double offset):

  file(file),
  particleIndices(particleIndices),
  signalForceWeights(signalForceWeights),
  scale(scale),
  offset(offset),
  usePeriodic(false)
  {
}

const string& PyTorchForceE2E::getFile() const {
  return file;
}
const double PyTorchForceE2E::getScale() const {
  return scale;
}
const double PyTorchForceE2E::getOffset() const {
  return offset;
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


int PyTorchForceE2E::addGlobalParameter(const string& name, double defaultValue) {
	globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
	return globalParameters.size()-1;
}

int PyTorchForceE2E::getNumGlobalParameters() const {
	return globalParameters.size();
}

const string& PyTorchForceE2E::getGlobalParameterName(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].name;
}

void PyTorchForceE2E::setGlobalParameterName(int index, const string& name) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].name = name;
}

double PyTorchForceE2E::getGlobalParameterDefaultValue(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].defaultValue;
}

void PyTorchForceE2E::setGlobalParameterDefaultValue(int index, double defaultValue) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].defaultValue = defaultValue;
}
