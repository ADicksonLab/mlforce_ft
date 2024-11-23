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

	std::vector<torch::Tensor> fixedInputs = force.getFixedInputs();
	// Assume:
	// fixedInputs[0] : time (float, 1)
	// fixedInputs[1] : sigma (float, 1)
	// fixedInputs[2] : atomtype (int, N)
	// fixedInputs[3] : edgeIndex (int, 2, Ne)
	// fixedInputs[4] : edgeType (int, Ne)
	// fixedInputs[5] : batch (int, N)

	node.setFloatProperty("time", fixedInputs[0].item<double>());
	node.setFloatProperty("sigma", fixedInputs[1].item<double>());

	int n_g = fixedInputs[2].sizes()[0];
	int n_e = fixedInputs[3].sizes()[1];
	node.setIntProperty("n_ghosts",n_g);
	node.setIntProperty("n_edges",n_e);
	
	SerializationNode&  atomTypeNode = node.createChildNode("AtomType");
	for (int i = 0; i < n_g; i++) {
	  atomTypeNode.createChildNode("type").setIntProperty("value", fixedInputs[2][i].item<int>());
	}

	SerializationNode& edgeIndexNode = node.createChildNode("EdgeIndex");
	for (int i = 0; i < 2; i++) {
		SerializationNode&  indexNode = targetfeaturesNode.createChildNode("Indexes");
		for (int j= 0; j < n_e; j++){
		  indexNode.createChildNode("index").setIntProperty("value", fixedInputs[3][i][j].item<int>);
		}
	}

	SerializationNode&  edgeTypeNode = node.createChildNode("EdgeType");
	for (int i = 0; i < n_e; i++) {
	  edgeTypeNode.createChildNode("type").setIntProperty("value", fixedInputs[4][i].item<int>());
	}

	SerializationNode&  BatchNode = node.createChildNode("Batch");
	for (int i = 0; i < n_g; i++) {
	  BatchNode.createChildNode("batch").setIntProperty("value", fixedInputs[5][i].item<int>());
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
	
	torch::Tensor time = torch::empty({1},torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor sigma = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor atomType = torch::empty({n_g}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor edgeIndex = torch::empty({2,n_e}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor edgeType = torch::empty({n_e}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor batch = torch::empty({n_g}, torch::TensorOptions().dtype(torch::kInt64));

	time[0] = node.getFloatProperty("time");
	sigma[0] = node.getFloatProperty("sigma");

	const SerializationNode& atomTypeNode = node.getChildNode("AtomType");
	int idx = 0;
	for (auto &type: atomTypeNode.getChildren()) {
	  atomType[idx] = type.getIntProperty("value");
	  idx += 1;
	}

	const SerializationNode& edgeIndexNode = node.getChildNode("EdgeIndex");
	idx = 0;
	for (auto &indexNode: edgeIndexNode.getChildren()) {
	  int idx2 = 0;
	  for (auto &index: indexNode.getChildren()) {
		edgeIndex[idx,idx2] = index.getIntProperty("value");
		idx2 += 1;
	  }
	  idx += 1;
	}
	
	const SerializationNode& edgeTypeNode = node.getChildNode("EdgeType");
	idx = 0;
	for (auto &type: edgeTypeNode.getChildren()) {
	  edgeType[idx] = type.getIntProperty("value");
	  idx += 1;
	}

	const SerializationNode& BatchNode = node.getChildNode("Batch");
	idx = 0;
	for (auto &batchnode: BatchNode.getChildren()) {
	  batch[idx] = batchnode.getIntProperty("value");
	  idx += 1;
	}

	std::vector<torch::Tensor> fixedInputs = {time, sigma, atomType, edgeIndex, edgeType, batch};
	
	PyTorchForceE2EDirect* force = new PyTorchForceE2EDirect(node.getStringProperty("file"),
															 indices,
															 signalForceWeights,
															 node.getDoubleProperty("scale"),
															 fixedInputs,
															 node.getBoolProperty("useAttr"));
												 
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
