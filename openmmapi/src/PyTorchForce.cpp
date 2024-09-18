#include "PyTorchForce.h"
#include "internal/PyTorchForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <vector>
using std::vector;

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForce::PyTorchForce(const std::string& file,
						   std::vector<std::vector<double> > targetFeatures,
						   const std::vector<int> particleIndices,
						   const std::vector<double> signalForceWeights,
						   const double scale,
						   const int assignFreq,
						   std::vector<std::vector<int> > restraintIndices, 
						   const std::vector<double> restraintDistances, 
						   const double rmaxDelta, 
						   const double restraintK,
						   const std::vector<int> initialAssignment) :

  file(file),
  targetFeatures(targetFeatures),
  particleIndices(particleIndices),
  signalForceWeights(signalForceWeights),
  scale(scale),
  assignFreq(assignFreq),
  usePeriodic(false),
  restraintIndices(restraintIndices),
  restraintDistances(restraintDistances),
  rmaxDelta(rmaxDelta),
  restraintK(restraintK),
  initialAssignment(initialAssignment)
  {
}

const string& PyTorchForce::getFile() const {
  return file;
}
const double PyTorchForce::getScale() const {
  return scale;
}
const int PyTorchForce::getAssignFreq() const {
  return assignFreq;
}

const std::vector<int> PyTorchForce::getInitialAssignment() const {
  return initialAssignment;
}

const std::vector<std::vector<int> > PyTorchForce::getRestraintIndices() const{
  return restraintIndices;
}

const std::vector<double> PyTorchForce::getRestraintDistances() const{
  return restraintDistances;
}

const std::vector<double> PyTorchForce::getRestraintParams() const{
	std::vector<double> params = {rmaxDelta, restraintK};
  	return params;	
}

const std::vector<std::vector<double> > PyTorchForce::getTargetFeatures() const{
  return targetFeatures;
}

const std::vector<int> PyTorchForce::getParticleIndices() const{
  return particleIndices;
}


const std::vector<double> PyTorchForce::getSignalForceWeights() const{
  return signalForceWeights;
}
ForceImpl* PyTorchForce::createImpl() const {
  return new PyTorchForceImpl(*this);
}

void PyTorchForce::setUsesPeriodicBoundaryConditions(bool periodic) {
	usePeriodic = periodic;
}

bool PyTorchForce::usesPeriodicBoundaryConditions() const {
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
