#ifndef REFERENCE_PY_TORCH_KERNELS_H_
#define REFERENCE_PY_TORCH_KERNELS_H_

#include "PyTorchKernels.h"
#include "Hungarian.h"
//#include "Distances.h"
#include "openmm/Platform.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstring>

using namespace std;
using namespace torch::indexing;
static const std::vector<string> PARAMETERNAMES={"charge_g", "sigma_g", "epsilon_g", "lambda_g"};

namespace PyTorchPlugin {

/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcPyTorchForceKernel : public CalcPyTorchForceKernel {
public:
	ReferenceCalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceKernel(name, platform) {
	}
	~ReferenceCalcPyTorchForceKernel();
      /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the PyTorchForce this kernel will be used for
     * @param module         the Pytorch model to use for computing forces and energy
     */
	void initialize(const OpenMM::System& system, const PyTorchForce& force,
			torch::jit::script::Module& nnModule);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
	double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
	torch::jit::script::Module nnModule;
	torch::Tensor boxVectorsTensor;
	torch::Tensor targetFeaturesTensor;
	std::vector<int> particleIndices;
    std::vector<double> signalForceWeights;
    std::vector<std::vector<double> > targetFeatures;
    std::vector<std::vector<int> > targetRestraintIndices;
    std::vector<double> targetRestraintDistances;
	std::vector<double> targetRestraintParams;
	std::vector<double> rmax, r0sq, restraint_b;
	double restraint_k, rmax_delta;
    double scale, offset;
	bool usePeriodic;
    HungarianAlgorithm hungAlg;
    int step_count;
    int assignFreq;
	int numRestraints;
    std::vector<int> assignment;
	std::vector<int> reverse_assignment;
};

class ReferenceCalcPyTorchForceE2EKernel : public CalcPyTorchForceE2EKernel {
public:
    ReferenceCalcPyTorchForceE2EKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceE2EKernel(name, platform) {
	}
	~ReferenceCalcPyTorchForceE2EKernel();
      /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the PyTorchForce this kernel will be used for
     * @param module         the Pytorch model to use for computing forces and energy
     */
	void initialize(const OpenMM::System& system, const PyTorchForceE2E& force,
			torch::jit::script::Module& nnModule);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
	double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
	torch::jit::script::Module nnModule;
	torch::Tensor boxVectorsTensor;
    torch::Tensor edge_idxs, edge_attrs, batch;
	std::vector<int> particleIndices;
    std::vector<double> signalForceWeights;
    double scale, offset;
	bool usePeriodic;
};

class ReferenceCalcPyTorchForceE2EDirectKernel : public CalcPyTorchForceE2EDirectKernel {
public:
    ReferenceCalcPyTorchForceE2EDirectKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceE2EDirectKernel(name, platform) {
	}
	~ReferenceCalcPyTorchForceE2EDirectKernel();
      /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the PyTorchForce this kernel will be used for
     * @param module         the Pytorch model to use for computing forces and energy
     */
	void initialize(const OpenMM::System& system, const PyTorchForceE2EDirect& force,
			torch::jit::script::Module& nnModule);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
	double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
  torch::jit::script::Module nnModule;
  torch::Tensor boxVectorsTensor;
  std::vector<torch::Tensor> fixedInputs;
  std::vector<int> particleIndices;
  std::vector<double> signalForceWeights;
  double scale;
  bool usePeriodic;
  bool useAttr;
  torch::TensorOptions options_float, options_int;

};
  
class ReferenceCalcPyTorchForceE2EDiffConfKernel : public CalcPyTorchForceE2EDiffConfKernel {
    public:
        ReferenceCalcPyTorchForceE2EDiffConfKernel(std::string name, const OpenMM::Platform& platform) : CalcPyTorchForceE2EDiffConfKernel(name, platform) {
        }
        ~ReferenceCalcPyTorchForceE2EDiffConfKernel();
          /**
         * Initialize the kernel.
         *
         * @param system         the System this kernel will be applied to
         * @param force          the PyTorchForce this kernel will be used for
         * @param module         the Pytorch model to use for computing forces and energy
         */
        void initialize(const OpenMM::System& system, const PyTorchForceE2EDiffConf& force,
                torch::jit::script::Module& nnModule);
    
        /**
         * Execute the kernel to calculate the forces and/or energy.
         *
         * @param context        the context in which to execute this kernel
         * @param includeForces  true if forces should be calculated
         * @param includeEnergy  true if the energy should be calculated
         * @return the potential energy due to the force
         */
        double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    private:
      torch::jit::script::Module nnModule;
      torch::Tensor boxVectorsTensor;
      std::vector<torch::Tensor> fixedInputs;
      std::vector<int> particleIndices;
      std::vector<double> signalForceWeights;
      double scale;
      bool usePeriodic;
      torch::TensorOptions options_float, options_int;
    
    };
} // namespace PyTorchPlugin

#endif /*REFERENCE_NEURAL_NETWORK_KERNELS_H_*/
