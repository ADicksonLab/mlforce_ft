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

PyTorchForceE2EDiffConfProxy::PyTorchForceE2EDiffConfProxy() : SerializationProxy("PyTorchForceE2EDiffConf") {
}

void PyTorchForceE2EDiffConfProxy::serialize(const void* object, SerializationNode& node) const {
	node.setIntProperty("version", 1);
	const PyTorchForceE2EDiffConf& force = *reinterpret_cast<const PyTorchForceE2EDiffConf*>(object);
	node.setStringProperty("file", force.getFile());
	node.setDoubleProperty("scale", force.getScale());
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

	std::vector<int> atoms = force.getAtomTypes();
	std::vector<std::vector<int>> bonds = force.getEdgeIndices();
	std::vector<std::vector<int>> angles = force.getAngles();
	std::vector<std::vector<int>> propers = force.getPropers();
	std::vector<std::vector<int>> impropers = force.getImpropers();
	std::vector<std::vector<int>> pairs = force.getPairs();
	std::vector<std::vector<int>> tetras = force.getTetras();
	std::vector<std::vector<int>> cistrans = force.getCisTrans();
	std::vector<std::vector<float>> encoding = force.getEncoding();

	SerializationNode&  atomTypeNode = node.createChildNode("AtomType");
	for (int i = 0; i < atoms.size(); i++) {
	  atomTypeNode.createChildNode("type").setIntProperty("value", atoms[i]);
	}

	SerializationNode& edgeIndexNode = node.createChildNode("EdgeIndex");
	for (int i = 0; i < bonds.size(); i++) {
		SerializationNode&  indexNode = edgeIndexNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0", bonds[i][0]);
		indexNode.setIntProperty("val1", bonds[i][1]);
	}

	SerializationNode& anglesNode = node.createChildNode("Angles");
	for (int i = 0; i < angles.size(); i++) {
		SerializationNode&  indexNode = anglesNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",angles[i][0]);
		indexNode.setIntProperty("val1",angles[i][1]);
		indexNode.setIntProperty("val2",angles[i][2]);
		indexNode.setIntProperty("val3",angles[i][3]);
	}

	SerializationNode& propersNode = node.createChildNode("Propers");
	for (int i = 0; i < propers.size(); i++) {
		SerializationNode&  indexNode = propersNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",propers[i][0]);
		indexNode.setIntProperty("val1",propers[i][1]);
		indexNode.setIntProperty("val2",propers[i][2]);
		indexNode.setIntProperty("val3",propers[i][3]);
		indexNode.setIntProperty("val4",propers[i][4]);
	}

	SerializationNode& impropersNode = node.createChildNode("Impropers");
	for (int i = 0; i < impropers.size(); i++) {
		SerializationNode&  indexNode = impropersNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",impropers[i][0]);
		indexNode.setIntProperty("val1",impropers[i][1]);
		indexNode.setIntProperty("val2",impropers[i][2]);
		indexNode.setIntProperty("val3",impropers[i][3]);
		indexNode.setIntProperty("val4",impropers[i][4]);
	}

	SerializationNode& pairsNode = node.createChildNode("Pairs");
	for (int i = 0; i < pairs.size(); i++) {
		SerializationNode&  indexNode = pairsNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",pairs[i][0]);
		indexNode.setIntProperty("val1",pairs[i][1]);
	}

	SerializationNode& tetrasNode = node.createChildNode("Tetras");
	for (int i = 0; i < tetras.size(); i++) {
		SerializationNode&  indexNode = tetrasNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",tetras[i][0]);
		indexNode.setIntProperty("val1",tetras[i][1]);
		indexNode.setIntProperty("val2",tetras[i][2]);
		indexNode.setIntProperty("val3",tetras[i][3]);
		indexNode.setIntProperty("val4",tetras[i][4]);
	}

	SerializationNode& cistransNode = node.createChildNode("CisTrans");
	for (int i = 0; i < cistrans.size(); i++) {
		SerializationNode&  indexNode = cistransNode.createChildNode("Indexes");
		indexNode.setIntProperty("val0",cistrans[i][0]);
		indexNode.setIntProperty("val1",cistrans[i][1]);
		indexNode.setIntProperty("val2",cistrans[i][2]);
		indexNode.setIntProperty("val3",cistrans[i][3]);
		indexNode.setIntProperty("val4",cistrans[i][4]);
	}	

	SerializationNode& encodingNode = node.createChildNode("Encoding");
	for (int i = 0; i < encoding.size(); i++) {
		SerializationNode&  indexNode = encodingNode.createChildNode("Indexes");
		for (int j = 0; j < encoding[0].size(); j++) {
		  indexNode.createChildNode("Value").setDoubleProperty("value",double(encoding[i][j]));
		}
	}
}

void* PyTorchForceE2EDiffConfProxy::deserialize(const SerializationNode& node) const {
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
	
	std::vector<int> atoms;
	std::vector<std::vector<int>> bonds;
	std::vector<std::vector<int>> angles;
	std::vector<std::vector<int>> propers;
	std::vector<std::vector<int>> impropers;
	std::vector<std::vector<int>> pairs;
	std::vector<std::vector<int>> tetras;
	std::vector<std::vector<int>> cistrans;
	std::vector<std::vector<float>> encoding;
	
	const SerializationNode& atomTypeNode = node.getChildNode("AtomType");
	for (auto &type: atomTypeNode.getChildren()) {
	  atoms.push_back(type.getIntProperty("value"));
	}

	const SerializationNode& edgeIndexNode = node.getChildNode("EdgeIndex");
	for (auto &indexNode: edgeIndexNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		bonds.push_back(tmp);
	}

	const SerializationNode& anglesNode = node.getChildNode("Angles");
	for (auto &indexNode: anglesNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		tmp.push_back(indexNode.getIntProperty("val2"));
		tmp.push_back(indexNode.getIntProperty("val3"));
		angles.push_back(tmp);
	}

	const SerializationNode& propersNode = node.getChildNode("Propers");
	for (auto &indexNode: propersNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		tmp.push_back(indexNode.getIntProperty("val2"));
		tmp.push_back(indexNode.getIntProperty("val3"));
		tmp.push_back(indexNode.getIntProperty("val4"));
		propers.push_back(tmp);
	}

	const SerializationNode& impropersNode = node.getChildNode("Impropers");
	for (auto &indexNode: impropersNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		tmp.push_back(indexNode.getIntProperty("val2"));
		tmp.push_back(indexNode.getIntProperty("val3"));
		tmp.push_back(indexNode.getIntProperty("val4"));
		impropers.push_back(tmp);
	}

	const SerializationNode& pairsNode = node.getChildNode("Pairs");
	for (auto &indexNode: pairsNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		pairs.push_back(tmp);
	}

	const SerializationNode& tetrasNode = node.getChildNode("Tetras");
	for (auto &indexNode: tetrasNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		tmp.push_back(indexNode.getIntProperty("val2"));
		tmp.push_back(indexNode.getIntProperty("val3"));
		tmp.push_back(indexNode.getIntProperty("val4"));
		tetras.push_back(tmp);
	}

	const SerializationNode& cistransNode = node.getChildNode("CisTrans");
	for (auto &indexNode: cistransNode.getChildren()) {
	  	std::vector<int> tmp;
		tmp.push_back(indexNode.getIntProperty("val0"));
		tmp.push_back(indexNode.getIntProperty("val1"));
		tmp.push_back(indexNode.getIntProperty("val2"));
		tmp.push_back(indexNode.getIntProperty("val3"));
		tmp.push_back(indexNode.getIntProperty("val4"));
		cistrans.push_back(tmp);
	}

	const SerializationNode& encodingNode = node.getChildNode("Encoding");
	for (auto &indexNode: encodingNode.getChildren()) {
	  	std::vector<float> tmp;
		for (auto &valueNode: indexNode.getChildren()) {
		  tmp.push_back(float(valueNode.getDoubleProperty("value")));
		}
		encoding.push_back(tmp);
	}
	
	PyTorchForceE2EDiffConf* force = new PyTorchForceE2EDiffConf(node.getStringProperty("file"),
															 indices,
															 signalForceWeights,
															 node.getDoubleProperty("scale"),
															 atoms,
															 bonds,
															 angles,
															 propers,
															 impropers,
															 pairs,
															 tetras,
															 cistrans,
															 encoding);
												 
	 if (node.hasProperty("forceGroup"))
	   force->setForceGroup(node.getIntProperty("forceGroup", 0));

	 if (node.hasProperty("usesPeriodic"))
	   force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));

	return force;
}
