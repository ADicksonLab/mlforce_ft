#ifndef PY_TORCH_KERNELS_H_
#define PY_TORCH_KERNELS_H_

#include "PyTorchForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
//#include <c_api.h> tensorflow
#include <string>

namespace PyTorchPlugin {

/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPyTorchForceKernel : public OpenMM::KernelImpl {
public:
	static std::string Name() {
	return "CalcPyTorchForce";
	}
	CalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
	}
	/**
	 * Initialize the kernel.
	 *
	 * @param system         the System this kernel will be applied to
	 * @param force          the PyTorchForce this kernel will be used for
	 * @param module         the Pytorch model to use for computing forces and energy
	 */
  virtual void initialize(const OpenMM::System& system, const PyTorchForce& force,
						  torch::jit::script::Module nnModule) = 0;
						  // std::vector<std::vector<double>> initialSignals,
						  // std::vector<std::vector<double>> targetFeatures) = 0;
  //at::ScalarType positionsType, at::ScalarType boxType, at::ScalarType energyType,
  //			  at::ScalarTypeforcesType) = 0;
  /**
	 * Execute the kernel to calculate the forces and/or energy.
	 *
	 * @param context        the context in which to execute this kernel
	 * @param includeForces  true if forces should be calculated
	 * @param includeEnergy  true if the energy should be calculated
	 * @return the potential energy due to the force
	 */
	virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};


/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPyTorchForceE2EKernel : public OpenMM::KernelImpl {
public:
	static std::string Name() {
	return "CalcPyTorchForceE2E";
	}
	CalcPyTorchForceE2EKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
	}
	/**
	 * Initialize the kernel.
	 *
	 * @param system         the System this kernel will be applied to
	 * @param force          the PyTorchForce this kernel will be used for
	 * @param module         the Pytorch model to use for computing forces and energy
	 */
  virtual void initialize(const OpenMM::System& system, const PyTorchForceE2E& force,
						  torch::jit::script::Module nnModule) = 0;
						  // std::vector<std::vector<double>> initialSignals,
						  // std::vector<std::vector<double>> targetFeatures) = 0;
  //at::ScalarType positionsType, at::ScalarType boxType, at::ScalarType energyType,
  //			  at::ScalarTypeforcesType) = 0;
  /**
	 * Execute the kernel to calculate the forces and/or energy.
	 *
	 * @param context        the context in which to execute this kernel
	 * @param includeForces  true if forces should be calculated
	 * @param includeEnergy  true if the energy should be calculated
	 * @return the potential energy due to the force
	 */
	virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

  
} // namespace PyTorchPlugin

#endif /*PY_TORCH_KERNELS_H_*/
