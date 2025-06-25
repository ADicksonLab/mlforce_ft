#include "CudaPyTorchKernels.h"
#include "CudaPyTorchKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>
#include <fstream>
#include <assert.h>

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
	vector<int> tmpAtomTypes = force.getAtomTypes();
	vector<vector<int>> tmpEdgeIdxs = force.getEdgeIndices();
	vector<int> tmpEdgeTypes = force.getEdgeTypes();
	useAttr = force.getUseAttr();
	usePeriodic = force.usesPeriodicBoundaryConditions();
	
	int n_edges = tmpEdgeTypes.size();
	int numGhostParticles = particleIndices.size();
	assert(tmpAtomTypes.size() == numGhostParticles);
	assert(tmpEdgeIdxs.size() == 2);
	assert(tmpEdgeIdxs[0].size() == n_edges);
	assert(tmpEdgeIdxs[1].size() == n_edges);

	options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
	options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

	// define tensors used for model inputs
	torch::Tensor atom_types_tensor = torch::empty({static_cast<int64_t>(numGhostParticles)}, options_int);
	auto at_acc = atom_types_tensor.accessor<int64_t, 1>();
	
	//Copy data to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  at_acc[i] = tmpAtomTypes[i];
	}

	torch::Tensor edge_idxs_tensor = torch::empty({2, static_cast<int64_t>(n_edges)}, options_int);
	auto edge_acc = edge_idxs_tensor.accessor<int64_t, 2>();
	torch::Tensor edge_types_tensor = torch::empty({static_cast<int64_t>(n_edges)}, options_int);
	auto et_acc = edge_types_tensor.accessor<int64_t, 1>();

	//Copy data to the tensors
	for (int i = 0; i < n_edges; i++) {
	  edge_acc[0][i] = tmpEdgeIdxs[0][i];
	  edge_acc[1][i] = tmpEdgeIdxs[1][i];
	  et_acc[i] = tmpEdgeTypes[i];
	}

	torch::Tensor batch = torch::zeros({numGhostParticles}, options_int);

	//                             |------------- fixedInputs ---------|
	// nnInputs = {[attr], pos, t, atomTypes, edgeIdxs, edgeTypes, batch, useGlobal}
	fixedInputs = {atom_types_tensor, edge_idxs_tensor, edge_types_tensor, batch};
	
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

	double useGlobal = context.getParameter("use_global");
	//torch::Tensor useGlobalTensor = torch::Tensor({useGlobal},options_float);
	torch::Tensor useGlobalTensor = torch::ones({1}, options_float) * useGlobal;

    vector<Vec3> MDPositions;
    context.getPositions(MDPositions);

	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 2>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0] = MDPositions[particleIndices[i]][0] *10;
		positions[i][1] = MDPositions[particleIndices[i]][1] *10;
		positions[i][2] = MDPositions[particleIndices[i]][2] *10;
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

	torch::Tensor timTensor = torch::ones({1}, options_float);
	
	nnInputs.push_back(positionsTensor);
	nnInputs.push_back(timTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}
	nnInputs.push_back(useGlobalTensor);

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");

	auto get_diffusion_noise = nnModule.get_method("get_diffusion_noise");
	torch::Tensor noise = scale*get_diffusion_noise(nnInputs).toTensor();

	//std::cout << "tim_idx, sigfac, mean noise:" << tim_idx << " " << sigfac << " " << torch::pow(noise, 2).mean() << "\n";
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
