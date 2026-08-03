#include "PyTorchForce.h"
#include "internal/PyTorchForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <vector>
using std::vector;

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EDiffConf::PyTorchForceE2EDiffConf(const std::string& file,
											 const std::vector<int> particleIndices,
											 const std::vector<double> signalForceWeights,
											 const double scale,
											 const std::vector<int> atoms,
											 const std::vector<std::vector<int>> bonds,
											 const std::vector<std::vector<int>> angles,
											 const std::vector<std::vector<int>> propers,
											 const std::vector<std::vector<int>> impropers,
											 const std::vector<std::vector<int>> pairs,
											 const std::vector<std::vector<int>> tetras,
											 const std::vector<std::vector<int>> cistrans,
											 const std::vector<std::vector<float>> encoding
											 ):

  file(file),
  particleIndices(particleIndices),
  signalForceWeights(signalForceWeights),
  scale(scale),
  atoms(atoms),
  bonds(bonds),
  angles(angles),
  propers(propers),
  impropers(impropers),
  pairs(pairs),
  tetras(tetras),
  cistrans(cistrans),
  encoding(encoding),
  usePeriodic(false)
  {
}

const string& PyTorchForceE2EDiffConf::getFile() const {
  return file;
}
const double PyTorchForceE2EDiffConf::getScale() const {
  return scale;
}
const std::vector<int> PyTorchForceE2EDiffConf::getAtomTypes() const {
  return atoms;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getEdgeIndices() const {
  return bonds;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getAngles() const {
	return angles;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getPropers() const {
	return propers;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getImpropers() const {
	return impropers;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getPairs() const {
	return pairs;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getTetras() const {
	return tetras;
}
const std::vector<std::vector<int>> PyTorchForceE2EDiffConf::getCisTrans() const {
	return cistrans;
}
const std::vector<std::vector<float>> PyTorchForceE2EDiffConf::getEncoding() const {
	return encoding;
}

const std::vector<int> PyTorchForceE2EDiffConf::getParticleIndices() const{
  return particleIndices;
}
const std::vector<double> PyTorchForceE2EDiffConf::getSignalForceWeights() const{
  return signalForceWeights;
}
ForceImpl* PyTorchForceE2EDiffConf::createImpl() const {
  return new PyTorchForceE2EDiffConfImpl(*this);
}

void PyTorchForceE2EDiffConf::setUsesPeriodicBoundaryConditions(bool periodic) {
	usePeriodic = periodic;
}

bool PyTorchForceE2EDiffConf::usesPeriodicBoundaryConditions() const {
	return usePeriodic;
}

int PyTorchForceE2EDiffConf::addGlobalParameter(const string& name, double defaultValue) {
	globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
	return globalParameters.size()-1;
}

int PyTorchForceE2EDiffConf::getNumGlobalParameters() const {
	return globalParameters.size();
}

const string& PyTorchForceE2EDiffConf::getGlobalParameterName(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].name;
}

void PyTorchForceE2EDiffConf::setGlobalParameterName(int index, const string& name) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].name = name;
}

double PyTorchForceE2EDiffConf::getGlobalParameterDefaultValue(int index) const {
	ASSERT_VALID_INDEX(index, globalParameters);
	return globalParameters[index].defaultValue;
}

void PyTorchForceE2EDiffConf::setGlobalParameterDefaultValue(int index, double defaultValue) {
	ASSERT_VALID_INDEX(index, globalParameters);
	globalParameters[index].defaultValue = defaultValue;
}
