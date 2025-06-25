#include "PyTorchForceProxy.h"
#include "PyTorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <torch/script.h>
#include <string>
#include <sstream>
#include <iostream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceE2EDirectProxy::PyTorchForceE2EDirectProxy() : SerializationProxy("PyTorchForceE2EDirect") {
}

void PyTorchForceE2EDirectProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 1);
	const PyTorchForceE2EDirect& force = *reinterpret_cast<const PyTorchForceE2EDirect*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
	node.setBoolProperty("useAttr", force.getUseAttr());
	node.setIntProperty("forceGroup", force.getForceGroup());
	node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
	
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

	std::vector<int> atomTypes = force.getAtomTypes();
	std::vector<std::vector<int>> edgeIdxs = force.getEdgeIndices();
	std::vector<int> edgeTypes = force.getEdgeTypes();

	int n_g = atomTypes.size();
	int n_e = edgeTypes.size();
	node.setIntProperty("n_ghosts",n_g);
	node.setIntProperty("n_edges",n_e);
	
	SerializationNode&  atomTypeNode = node.createChildNode("AtomType");
	for (int i = 0; i < n_g; i++) {
	  atomTypeNode.createChildNode("type").setIntProperty("value", atomTypes[i]);
	}

	SerializationNode& edgeIndexNode = node.createChildNode("EdgeIndex");
	for (int i = 0; i < 2; i++) {
		SerializationNode&  indexNode = edgeIndexNode.createChildNode("Indexes");
		for (int j= 0; j < n_e; j++){
		  indexNode.createChildNode("index").setIntProperty("value", edgeIdxs[i][j]);
		}
	}

	SerializationNode&  edgeTypeNode = node.createChildNode("EdgeType");
	for (int i = 0; i < n_e; i++) {
	  edgeTypeNode.createChildNode("type").setIntProperty("value", edgeTypes[i]);
	}

}

void* PyTorchForceE2EDirectProxy::deserialize(const SerializationNode& node) const {
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

	int n_g = node.getIntProperty("n_ghosts");
	int n_e = node.getIntProperty("n_edges");
	
	std::vector<int> atomTypes;
	std::vector<std::vector<int>> edgeIdxs;
	std::vector<int> edgeTypes;
	
	const SerializationNode& atomTypeNode = node.getChildNode("AtomType");
	for (auto &type: atomTypeNode.getChildren()) {
	  atomTypes.push_back(type.getIntProperty("value"));
	}

	const SerializationNode& edgeIndexNode = node.getChildNode("EdgeIndex");
	for (auto &indexNode: edgeIndexNode.getChildren()) {
	  std::vector<int> tmp;
	  for (auto &index: indexNode.getChildren()) {
		tmp.push_back(index.getIntProperty("value"));
	  }
	  edgeIdxs.push_back(tmp);
	}
	
	const SerializationNode& edgeTypeNode = node.getChildNode("EdgeType");
	for (auto &type: edgeTypeNode.getChildren()) {
	  edgeTypes.push_back(type.getIntProperty("value"));
	}

	PyTorchForceE2EDirect* force = new PyTorchForceE2EDirect(node.getStringProperty("file"),
															 indices,
															 signalForceWeights,
															 node.getDoubleProperty("scale"),
															 atomTypes,
															 edgeIdxs,
															 edgeTypes,
															 node.getBoolProperty("useAttr"));
												 
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
