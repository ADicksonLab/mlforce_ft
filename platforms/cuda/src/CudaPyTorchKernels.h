#ifndef CUDA_PY_TORCH_KERNELS_H_
#define CUDA_PY_TORCH_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
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

#include "PyTorchKernels.h"
#include "Hungarian.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"


using namespace std;
using namespace torch::indexing;
static const std::vector<string> PARAMETERNAMES={"charge_g", "sigma_g", "epsilon_g", "lambda_g"};

namespace PyTorchPlugin {

/**
 * This kernel is invoked by PyTorchForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcPyTorchForceKernel : public CalcPyTorchForceKernel {
public:
    CudaCalcPyTorchForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
	    CalcPyTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcPyTorchForceKernel();
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
    OpenMM::CudaContext& cu;
    torch::jit::script::Module nnModule;
    torch::Tensor boxVectorsTensor, targetFeaturesTensor;
    std::vector<int> particleIndices;
    std::vector<double> signalForceWeights;
    std::vector<std::vector<double> > targetFeatures;
    std::vector<std::vector<int> > targetRestraintIndices;
    std::vector<double> targetRestraintDistances;
	std::vector<double> targetRestraintParams;
	std::vector<double> rmax, r0sq, restraint_b;
	double restraint_k, rmax_delta;
	double scale;
	bool usePeriodic;
    HungarianAlgorithm hungAlg;
    int step_count;
    int assignFreq;
    int numRestraints;
    std::vector<int> assignment;
    std::vector<int> reverse_assignment;
    CUfunction copyInputsKernel, addForcesKernel;
};

} // namespace PyTorchPlugin

#endif /*CUDA_PY_TORCH_KERNELS_H_*/
