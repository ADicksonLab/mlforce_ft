/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "PyTorchForceProxy.h"
#include "PyTorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

using std::vector;

using namespace PyTorchPlugin;
using namespace OpenMM;

PyTorchForceProxy::PyTorchForceProxy() : SerializationProxy("PyTorchForce") {
}

void PyTorchForceProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 2);
	const PyTorchForce& force = *reinterpret_cast<const PyTorchForce*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
	node.setDoubleProperty("lambdaPenalty", force.getLambdaMismatchPenalty());
	node.setIntProperty("assignFreq", force.getAssignFreq());
	node.setIntProperty("forceGroup", force.getForceGroup());
	node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
	std::pair<double,double> params = force.getRestraintParams();
	node.setDoubleProperty("rmaxDelta", params.first);
	node.setDoubleProperty("restraintK", params.second);

	vector<vector<vector<double>>> features = force.getTargetFeatures();
	SerializationNode& targetfeaturesNode = node.createChildNode("TargetFeatures");
	for (int i = 0; i < features.size(); i++) {
	  SerializationNode&  targetNode = targetfeaturesNode.createChildNode("Target");
	  for (int j= 0; j < features[i].size(); j++){
		SerializationNode&  atomNode = targetNode.createChildNode("Atom");
		for (int k= 0; k < features[i][j].size(); k++){
		  atomNode.createChildNode("feature").setDoubleProperty("value", features[i][j][k]);
		}
	  }
	}

	auto rest_data = force.getRestraintData();
	vector<vector<double>> targetRestraintDistances = rest_data.first;
	vector<vector<vector<int> >> targetRestraintIndices = rest_data.second;
	
	SerializationNode& restraintDataNode = node.createChildNode("RestraintData");
	for (int i = 0; i < targetRestraintDistances.size(); i++) {
	  SerializationNode&  targetNode = restraintDataNode.createChildNode("Target");
	  for (int j= 0; j < targetRestraintDistances[i].size(); j++){
		SerializationNode&  restNode = targetNode.createChildNode("Restraint");
		restNode.setDoubleProperty("distance",targetRestraintDistances[i][j]);
		restNode.setIntProperty("idx1",targetRestraintIndices[i][j][0]);
		restNode.setIntProperty("idx2",targetRestraintIndices[i][j][1]);
	  }
	}

	vector<int>  ParticleIndices = force.getParticleIndices();
	SerializationNode& ParticleIndicesNode = node.createChildNode("ParticleIndices");
	for (int i = 0; i < ParticleIndices.size(); i++) {
		 ParticleIndicesNode.createChildNode("Index").setIntProperty("value", ParticleIndices[i]);
	}

	std::pair<int,vector<int>> data = force.getInitialAssignment();
	int initial_idx = data.first;
	vector<int> initialAssignment = data.second;
	node.setIntProperty("initialIdx", initial_idx);

	SerializationNode& initialAssignmentNode = node.createChildNode("InitialAssignment");
	for (int i = 0; i < initialAssignment.size(); i++) {
	  initialAssignmentNode.createChildNode("assignment").setIntProperty("value", initialAssignment[i]);
	}
	
	vector<double>  signalForceWeights = force.getSignalForceWeights();
	SerializationNode&  signalForceWeightsNode = node.createChildNode("SignalForceWeights");
	for (int i = 0; i < signalForceWeights.size(); i++) {
	   signalForceWeightsNode.createChildNode("Weight").setDoubleProperty("value", signalForceWeights[i]);
	}

}

void* PyTorchForceProxy::deserialize(const SerializationNode& node) const {
	if (node.getIntProperty("version") != 2)
	throw OpenMMException("Unsupported version number");

	vector<vector<vector<double>>> targetfeatures;
	const SerializationNode& targetfeaturesNode = node.getChildNode("TargetFeatures");
	auto targetnodes = targetfeaturesNode.getChildren();
	int numTargets = targetnodes.size();
	for (int i=0; i<numTargets; i++){
	  auto atomnodes = targetnodes[i].getChildren();
	  int numAtoms = atomnodes.size();

	  vector<vector<double>> tmp_tgt_features(numAtoms);
	  for (int j=0; j<numAtoms; j++){
		for (auto &feature:atomnodes[j].getChildren()){
		  tmp_tgt_features[j].push_back(feature.getDoubleProperty("value"));
		}
	  }
	  targetfeatures.push_back(tmp_tgt_features);	
	}
	

	vector<vector<vector<int>>> restraint_idxs(numTargets);
	vector<vector<double>> restraint_dists(numTargets);

	const SerializationNode& restraintDataNode = node.getChildNode("RestraintData");
	auto restraintTgtNodes = restraintDataNode.getChildren();
	assert(numTargets == restraintTgtNodes.size());
	for (int i=0; i<numTargets; i++){
	  for (auto &rest:restraintTgtNodes[i].getChildren()){
		vector<int> idxs {rest.getIntProperty("idx1"), rest.getIntProperty("idx2")};
		restraint_idxs[i].push_back(idxs);
		restraint_dists[i].push_back(rest.getDoubleProperty("distance"));
	  }
	}

	vector<int> indices;
	const SerializationNode& partilceindicesNode = node.getChildNode("ParticleIndices");
	for (auto & index: partilceindicesNode.getChildren()) {
		indices.push_back(index.getIntProperty("value"));
	}

	vector<int> initialAssignment;
	const SerializationNode& initialAssignmentNode = node.getChildNode("InitialAssignment");
	for (auto & assignment: initialAssignmentNode.getChildren()) {
		initialAssignment.push_back(assignment.getIntProperty("value"));
	}
	
	vector<double> signalForceWeights;
	const SerializationNode& signalForceWeightsNode = node.getChildNode("SignalForceWeights");
	for (auto &weight:signalForceWeightsNode.getChildren()){
		signalForceWeights.push_back(weight.getDoubleProperty("value"));
	}

	PyTorchForce* force = new PyTorchForce(node.getStringProperty("file"),  targetfeatures,
										   indices, signalForceWeights, node.getDoubleProperty("scale"), node.getIntProperty("assignFreq"),
										   restraint_idxs, restraint_dists, node.getDoubleProperty("rmaxDelta"), node.getDoubleProperty("restraintK"),
										   initialAssignment, node.getIntProperty("initialIdx"), node.getDoubleProperty("lambdaPenalty"));
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
