#ifndef OPENMM_PY_TORCH_FORCE_IMPL_H_
#define OPENMM_PY_TORCH_FORCE_IMPL_H_

#include "PyTorchForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <utility>
#include <set>
#include <string>

namespace PyTorchPlugin {

class System;

/**
 * This is the internal implementation of PyTorchForce.
 */

class OPENMM_EXPORT_PYTORCH PyTorchForceImpl : public OpenMM::ForceImpl {
public:
	PyTorchForceImpl(const PyTorchForce& owner);
	~PyTorchForceImpl();
	void initialize(OpenMM::ContextImpl& context);
	const PyTorchForce& getOwner() const {
	return owner;
	}
	void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
	// This force field doesn't update the state directly.
	}
	double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
	std::map<std::string, double> getDefaultParameters() {
	return std::map<std::string, double>(); // This force field doesn't define any parameters.
	}
	std::vector<std::string> getKernelNames();
private:
	const PyTorchForce& owner;
	OpenMM::Kernel kernel;
	torch::jit::script::Module nnModule;
	std::vector<std::vector<double>> targetFeatures;
	std::vector<int> particleIndicies;
};

/**
 * This is the internal implementation of PyTorchForceE2E.
 */

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EImpl : public OpenMM::ForceImpl {
public:
	PyTorchForceE2EImpl(const PyTorchForceE2E& owner);
	~PyTorchForceE2EImpl();
	void initialize(OpenMM::ContextImpl& context);
	const PyTorchForceE2E& getOwner() const {
	return owner;
	}
	void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
	// This force field doesn't update the state directly.
	}
	double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
	std::map<std::string, double> getDefaultParameters() {
	return std::map<std::string, double>(); // This force field doesn't define any parameters.
	}
	std::vector<std::string> getKernelNames();
private:
	const PyTorchForceE2E& owner;
	OpenMM::Kernel kernel;
	torch::jit::script::Module nnModule;
	std::vector<int> particleIndicies;
};

/**
 * This is the internal implementation of PyTorchForceE2EDirect.
 */

class OPENMM_EXPORT_PYTORCH PyTorchForceE2EDirectImpl : public OpenMM::ForceImpl {
public:
	PyTorchForceE2EDirectImpl(const PyTorchForceE2EDirect& owner);
	~PyTorchForceE2EDirectImpl();
	void initialize(OpenMM::ContextImpl& context);
	const PyTorchForceE2EDirect& getOwner() const {
	return owner;
	}
	void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
	// This force field doesn't update the state directly.
	}
	double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
	std::map<std::string, double> getDefaultParameters() {
	return std::map<std::string, double>(); // This force field doesn't define any parameters.
	}
	std::vector<std::string> getKernelNames();
private:
	const PyTorchForceE2EDirect& owner;
	OpenMM::Kernel kernel;
	torch::jit::script::Module nnModule;
	std::vector<int> particleIndicies;
};

  
} // namespace PyTorchPlugin

#endif /*OPENMM_PY_TORCH_FORCE_IMPL_H_*/
