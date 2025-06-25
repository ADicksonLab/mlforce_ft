#include "PyTorchForceProxy.h"
#include "PyTorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EProxy::PyTorchForceE2EProxy() : SerializationProxy("PyTorchForceE2E") {
}

void PyTorchForceE2EProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 1);
	const PyTorchForceE2E& force = *reinterpret_cast<const PyTorchForceE2E*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
	node.setDoubleProperty("offset", force.getOffset());
	node.setIntProperty("forceGroup", force.getForceGroup());
	node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
	node.setBoolProperty("usesLambda", force.usesLambda());

	std::vector<int>  ParticleIndices = force.getParticleIndices();
	SerializationNode& ParticleIndicesNode = node.createChildNode("ParticleIndices");
	for (int i = 0; i < ParticleIndices.size(); i++) {
		 ParticleIndicesNode.createChildNode("Index").setIntProperty("value", ParticleIndices[i]);
	}

	std::vector<double>  signalForceWeights = force.getSignalForceWeights();
	SerializationNode&  signalForceWeightsNode = node.createChildNode("SignalForceWeights");
	for (int i = 0; i < signalForceWeights.size(); i++) {
	   signalForceWeightsNode.createChildNode("Weight").setDoubleProperty("value", signalForceWeights[i]);
	}
}

void* PyTorchForceE2EProxy::deserialize(const SerializationNode& node) const {
	if (node.getIntProperty("version") != 1)
	throw OpenMMException("Unsupported version number");

	std::vector<int> indices;
	const SerializationNode& partilceindicesNode = node.getChildNode("ParticleIndices");
	for (auto & index: partilceindicesNode.getChildren()) {
		indices.push_back(index.getIntProperty("value"));
	}

	std::vector<double> signalForceWeights;
	const SerializationNode& signalForceWeightsNode = node.getChildNode("SignalForceWeights");
	for (auto &weight:signalForceWeightsNode.getChildren()){
		signalForceWeights.push_back(weight.getDoubleProperty("value"));
	}

	PyTorchForceE2E* force = new PyTorchForceE2E(node.getStringProperty("file"),
												 indices,
												 signalForceWeights,
												 node.getDoubleProperty("scale"),
												 node.getDoubleProperty("offset"),
												 node.getBoolProperty("usesLambda"));
												 
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
