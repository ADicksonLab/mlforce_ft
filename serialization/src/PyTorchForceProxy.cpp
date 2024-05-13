#include "PyTorchForceProxy.h"
#include "PyTorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

PyTorchForceProxy::PyTorchForceProxy() : SerializationProxy("PyTorchForce") {
}

void PyTorchForceProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 1);
	const PyTorchForce& force = *reinterpret_cast<const PyTorchForce*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
	node.setIntProperty("assignFreq", force.getAssignFreq());
	node.setIntProperty("forceGroup", force.getForceGroup());
	node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
	std::vector<double> params = force.getRestraintParams();
	node.setDoubleProperty("rmaxDelta", params[0]);
	node.setDoubleProperty("restraintK", params[1]);

	std::vector<vector<double>> features = force.getTargetFeatures();
	SerializationNode& targetfeaturesNode = node.createChildNode("TargetFeatures");
	for (int i = 0; i < features.size(); i++) {
		SerializationNode&  featureNode = targetfeaturesNode.createChildNode("Particles");
		for (int j= 0; j < features[0].size(); j++){
			featureNode.createChildNode("features").setDoubleProperty("value", features[i][j]);
		}
	}

	std::vector<vector<int>> rest_idxs = force.getRestraintIndices();
	SerializationNode& restraintIndicesNode = node.createChildNode("RestraintIndices");
	for (int i = 0; i < rest_idxs.size(); i++) {
		SerializationNode&  idxsNode = restraintIndicesNode.createChildNode("restraint");
		for (int j= 0; j < rest_idxs[0].size(); j++){
			idxsNode.createChildNode("idxs").setIntProperty("value", rest_idxs[i][j]);
		}
	}

	std::vector<int>  ParticleIndices = force.getParticleIndices();
	SerializationNode& ParticleIndicesNode = node.createChildNode("ParticleIndices");
	for (int i = 0; i < ParticleIndices.size(); i++) {
		 ParticleIndicesNode.createChildNode("Index").setIntProperty("value", ParticleIndices[i]);
	}

	std::vector<int>  initialAssignment = force.getInitialAssignment();
	SerializationNode& initialAssignmentNode = node.createChildNode("InitialAssignment");
	for (int i = 0; i < initialAssignment.size(); i++) {
		 initialAssignmentNode.createChildNode("assignment").setIntProperty("value", initialAssignment[i]);
	}
	
	std::vector<double>  signalForceWeights = force.getSignalForceWeights();
	SerializationNode&  signalForceWeightsNode = node.createChildNode("SignalForceWeights");
	for (int i = 0; i < signalForceWeights.size(); i++) {
	   signalForceWeightsNode.createChildNode("Weight").setDoubleProperty("value", signalForceWeights[i]);
	}

	std::vector<double>  rest_dists = force.getRestraintDistances();
	SerializationNode&  restraintDistancesNode = node.createChildNode("RestraintDistances");
	for (int i = 0; i < rest_dists.size(); i++) {
	   restraintDistancesNode.createChildNode("Distance").setDoubleProperty("value", rest_dists[i]);
	}
}

void* PyTorchForceProxy::deserialize(const SerializationNode& node) const {
	if (node.getIntProperty("version") != 1)
	throw OpenMMException("Unsupported version number");

	const SerializationNode& targetfeaturesNode = node.getChildNode("TargetFeatures");
	int	numTargetParticles = targetfeaturesNode.getChildren().size();

	std::vector<std::vector<double>> targetfeatures(numTargetParticles);
	for (int i=0; i<numTargetParticles; i++){
		const SerializationNode& featureNode = targetfeaturesNode.getChildren()[i];
		for (auto &feature:featureNode.getChildren()){
			targetfeatures[i].push_back(feature.getDoubleProperty("value"));
		}
	}

	const SerializationNode& restraintIndicesNode = node.getChildNode("RestraintIndices");
	int	numRestraints = restraintIndicesNode.getChildren().size();
	std::vector<std::vector<int>> restraint_idxs(numRestraints);
	for (int i=0; i<numRestraints; i++){
		const SerializationNode& idxsNode = restraintIndicesNode.getChildren()[i];
		for (auto &idx:idxsNode.getChildren()){
			restraint_idxs[i].push_back(idx.getIntProperty("value"));
		}
	}

	std::vector<double> restraint_dists;
	const SerializationNode& restraintDistancesNode = node.getChildNode("RestraintDistances");
	for (auto &distance:restraintDistancesNode.getChildren()){
		restraint_dists.push_back(distance.getDoubleProperty("value"));
	}

	std::vector<int> indices;
	const SerializationNode& partilceindicesNode = node.getChildNode("ParticleIndices");
	for (auto & index: partilceindicesNode.getChildren()) {
		indices.push_back(index.getIntProperty("value"));
	}

	std::vector<int> initialAssignment;
	const SerializationNode& initialAssignmentNode = node.getChildNode("InitialAssignment");
	for (auto & assignment: initialAssignmentNode.getChildren()) {
		initialAssignment.push_back(assignment.getIntProperty("value"));
	}
	
	std::vector<double> signalForceWeights;
	const SerializationNode& signalForceWeightsNode = node.getChildNode("SignalForceWeights");
	for (auto &weight:signalForceWeightsNode.getChildren()){
		signalForceWeights.push_back(weight.getDoubleProperty("value"));
	}

	PyTorchForce* force = new PyTorchForce(node.getStringProperty("file"),  targetfeatures,
										   indices, signalForceWeights, node.getDoubleProperty("scale"), node.getIntProperty("assignFreq"),
										   restraint_idxs, restraint_dists, node.getDoubleProperty("rmaxDelta"), node.getDoubleProperty("restraintK"),
										   initialAssignment);
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
