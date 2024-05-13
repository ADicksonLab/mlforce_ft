#ifndef OPENCL_PY_TORCH_KERNELS_H_
#define OPENCL_PY_TORCH_KERNELS_H_

#include "PyTorchKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "Hungarian.h"

using namespace std;
using namespace torch::indexing;
static const std::vector<string> PARAMETERNAMES={"charge_g", "sigma_g", "epsilon_g", "lambda_g"};

namespace PyTorchPlugin {

/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class OpenCLCalcPyTorchForceKernel : public CalcPyTorchForceKernel {
public:
    OpenCLCalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::OpenCLContext& cl) :
	    CalcPyTorchForceKernel(name, platform), hasInitializedKernel(false), cl(cl){
    }
    ~OpenCLCalcPyTorchForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the PyTorchForce this kernel will be used for
     * @param module         the Pytorch model to use for computing forces and energy
     */
    void initialize(const OpenMM::System& system, const PyTorchForce& force,
					torch::jit::script::Module nnModule);

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
    bool hasInitializedKernel;
    OpenMM::OpenCLContext& cl;
    torch::jit::script::Module nnModule;
	torch::Tensor boxVectorsTensor, targetFeaturesTensor;
    std::vector<int> particleIndices;
    std::vector<double> signalForceWeights;
    std::vector<std::vector<double>> targetFeatures;
    double scale;
    int assignFreq;
    int step_count;
    vector<int> assignment;
    bool usePeriodic;
    OpenMM::OpenCLArray networkForces;
    cl::Kernel addForcesKernel;
    HungarianAlgorithm hungAlg;
};

} // namespace NNPlugin

#endif /*OPENCL_PY_TORCH_KERNELS_H_*/
