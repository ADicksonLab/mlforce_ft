#include "CudaPyTorchKernels.h"
#include "CudaPyTorchKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>
#include <fstream>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;


/**
 * @brief
 *
 * @param context
 * @param numParticles
 * @return std::vector<double>
 */

// This function extracts and collects context variables for each particle and returns them as a vector of doubles.
// signals vector would have a length of numParticles * PARAMETERNAMES.size()

static std::vector<double> extractContextVariables(ContextImpl& context, int numParticles) {
	std::vector<double> signals;
	string name;
	for (int i=0; i < numParticles; i++) {
		for (std::size_t j=0; j < PARAMETERNAMES.size(); j++) {
			signals.push_back(context.getParameter(PARAMETERNAMES[j]+std::to_string(i)));
		}
	}
	return signals;
}

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
// report errors that occur during synchronization operations.
#define CHECK_RESULT(result, prefix) \
if (result != CUDA_SUCCESS) { \
	std::stringstream m; \
	m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
	throw OpenMMException(m.str());\
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::CudaContext& cu, torch::Tensor& tensor) {
    void* data;
    if (cu.getUseDoublePrecision()) {
        data = tensor.to(torch::kFloat64).data_ptr<double>();
    } else {
        data = tensor.to(torch::kFloat32).data_ptr<float>();
    }
    return data;
}

CudaCalcPyTorchForceE2EDirectKernel::CudaCalcPyTorchForceE2EDirectKernel(string name, const Platform& platform, CudaContext& cu): CalcPyTorchForceE2EDirectKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

/**
 * @brief Destroy the Cuda CalcPy Torch Force Kernel:: Cuda CalcPy Torch Force Kernel object
 *
 */
CudaCalcPyTorchForceE2EDirectKernel::~CudaCalcPyTorchForceE2EDirectKernel() {
  cuDevicePrimaryCtxRelease(cu.getDevice());
}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */

void CudaCalcPyTorchForceE2EDirectKernel::initialize(const System& system, const PyTorchForceE2EDirect& force, torch::jit::script::Module& nnModule) {

	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	vector<torch::Tensor> tmpFixedInputs = force.getFixedInputs();
	useAttr = force.getUseAttr();

	double beta_start = 1.0e-7;
	double beta_end = 2.0e-3;
	num_diff_steps = 100;

	sigma = {};
	double alpha = 1.0;
	for (int i = 0; i < num_diff_steps; i++) {
	  double tim = double(i)/double(num_diff_steps);
	  double beta = beta_start + (beta_end-beta_start)/(1.0 + exp(-tim));
	  alpha *= (1.0 - beta);
	  sigma.push_back(sqrt((1.0 - alpha)/alpha));
	}
	
	// 	Assume:
	// fixedInputs[0] : atomtype (int, N)
	// fixedInputs[1] : edgeIndex (int, 2, Ne)
	// fixedInputs[2] : edgeType (int, Ne)
	// fixedInputs[3] : batch (int, N)

	// nnInputs = {[attr], pos, t, *fixedInputs}

	
	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

	if (usePeriodic) {
	  int64_t boxVectorsDims[] = {3, 3};
	  boxVectorsTensor = torch::zeros(boxVectorsDims);
	  boxVectorsTensor = boxVectorsTensor.to(torch::kFloat32);
	}

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

	// Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initializes PyTorch
    //torch::TensorOptions options_gpu = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	// Pop the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

	torch::TensorOptions options_cpu = torch::TensorOptions().
	  device(torch::kCPU).
	  dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
	options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

	fixedInputs = {};
	fixedInputs.push_back(tmpFixedInputs[0].to(options_float));	
	fixedInputs.push_back(tmpFixedInputs[1].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[2].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[3].to(options_int));
	fixedInputs.push_back(tmpFixedInputs[4].to(options_int));

    ContextSelector selector(cu); // Switch to the OpenMM context
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaPyTorchKernelSources::PyTorchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
	
}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double CudaCalcPyTorchForceE2EDirectKernel::execute(ContextImpl& context,bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    int numParticles = cu.getNumAtoms();
	int numGhostParticles = particleIndices.size();

	// Get the current diffusion time from the context
	double tim = context.getParameter("diffTime");
	int tim_idx = int(floor(tim*num_diff_steps));
	if (tim_idx < 0) {
	  tim_idx = 0;
	} else if (tim_idx >= num_diff_steps) {
	  tim_idx = num_diff_steps - 1;
	}
 
	double sigfac = sigma[tim_idx]*0.01;
	
    vector<Vec3> MDPositions;
    context.getPositions(MDPositions);

	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 2>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0];
		positions[i][1] = MDPositions[particleIndices[i]][1];
		positions[i][2] = MDPositions[particleIndices[i]][2];
	}

	torch::Tensor signalsTensor = torch::empty({numGhostParticles, 4}, options_float.requires_grad(true));
	
	vector<torch::jit::IValue> nnInputs = {};
	if (useAttr) {
	  // if using signals, pass them first
	  std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);

	  auto signals = signalsTensor.accessor<float, 2>();
	  for (int i = 0; i < numGhostParticles; i++) {
		for (int j = 0; j < 4; j++) {
		  signals[i][j] = globalVariables[4*i + j];
		}
	  }
	
	  nnInputs.push_back(signalsTensor);
	}
	
	nnInputs.push_back(positionsTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");

	auto get_diffusion_noise = nnModule.get_method("get_diffusion_noise");
	torch::Tensor noise = sigfac*scale*get_diffusion_noise(nnInputs).toTensor();

	// get forces on positions as before
	if (includeForces) {

	  auto NNForce = noise.accessor<float, 2>();

	  if (useAttr) {
		// update the global variables derivatives
		map<string, double> &energyParamDerivs = cu.getEnergyParamDerivWorkspace();

		// saving signals derivatives to context
		if (cu.getUseDoublePrecision()) {
		  for (int i = 0; i < numGhostParticles; i++) {
			for (int j=0; j<4; j++){
			  energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += double(NNForce[i][j+3])*signalForceWeights[j];
			}
		  }
		} else {
		  for (int i = 0; i < numGhostParticles; i++) {
			for (int j=0; j<4; j++) {
			  energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += NNForce[i][j+3]*signalForceWeights[j];
			}
		  }		
		}
	  }
	  
	  // sending atomic forces to cuda context
	  torch::Tensor forceTensor = noise.index({torch::indexing::Slice(), torch::indexing::Slice(0,3)});  
	  torch::Tensor paddedForceTensor = torch::zeros({numParticles, 3});
	  paddedForceTensor.narrow(0,
							   static_cast<int64_t>(particleIndices[0]),
							   static_cast<int64_t>(particleIndices.size())).copy_(forceTensor);

	  torch::Device device(torch::kCUDA, cu.getDeviceIndex());
	  paddedForceTensor = paddedForceTensor.to(device);
	  void* fdata = getTensorPointer(cu, paddedForceTensor);
		
	  CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
	  {
		ContextSelector selector(cu);
		int paddedNumAtoms = cu.getPaddedNumAtoms();
		void* forceArgs[] = {&fdata, &cu.getForce().getDevicePointer(),
		  &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
		cu.executeKernel(addForcesKernel, forceArgs, numParticles);
		CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
	  }
	}

    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped

	return 0.0; // E2EDirect only updates forces, there is no energy

}
