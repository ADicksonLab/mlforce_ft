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

CudaCalcPyTorchForceE2EDiffConfKernel::CudaCalcPyTorchForceE2EDiffConfKernel(string name, const Platform& platform, CudaContext& cu): CalcPyTorchForceE2EDiffConfKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

/**
 * @brief Destroy the Cuda CalcPy Torch Force Kernel:: Cuda CalcPy Torch Force Kernel object
 *
 */
CudaCalcPyTorchForceE2EDiffConfKernel::~CudaCalcPyTorchForceE2EDiffConfKernel() {
  cuDevicePrimaryCtxRelease(cu.getDevice());
}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */

void CudaCalcPyTorchForceE2EDiffConfKernel::initialize(const System& system, const PyTorchForceE2EDiffConf& force, torch::jit::script::Module& nnModule) {

	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();

	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	vector<int> tmpAtomTypes = force.getAtomTypes();
	vector<vector<int>> tmpEdgeIdxs = force.getEdgeIndices();
	vector<vector<int>> tmpAngles = force.getAngles();
	vector<vector<int>> tmpPropers = force.getPropers();
	vector<vector<int>> tmpImpropers = force.getImpropers();
	vector<vector<int>> tmpPairs = force.getPairs();
	vector<vector<int>> tmpTetras = force.getTetras();
	vector<vector<int>> tmpCisTrans = force.getCisTrans();
	vector<vector<float>> tmpEncoding = force.getEncoding();
	usePeriodic = force.usesPeriodicBoundaryConditions();

		
	int n_edges = tmpEdgeIdxs.size();
	int numGhostParticles = particleIndices.size();
	assert(tmpAtomTypes.size() == numGhostParticles);
	assert(tmpEdgeIdxs[0].size() == 2);

	options_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
	options_int = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

	// define tensors used for model inputs
	torch::Tensor atoms_tensor = torch::empty({static_cast<int64_t>(numGhostParticles)}, options_int);
	auto at_acc = atoms_tensor.accessor<int64_t, 1>();

	torch::Tensor bonds_tensor = torch::empty({static_cast<int64_t>(n_edges), 2}, options_int);
	auto bonds_acc = bonds_tensor.accessor<int64_t, 2>();
	
	torch::Tensor angles_tensor = torch::empty({static_cast<int64_t>(tmpAngles.size()), 4}, options_int);
	auto angles_acc = angles_tensor.accessor<int64_t, 2>();

	torch::Tensor propers_tensor = torch::empty({static_cast<int64_t>(tmpPropers.size()), 5}, options_int);
	auto prop_acc = propers_tensor.accessor<int64_t, 2>();

	torch::Tensor impropers_tensor = torch::empty({static_cast<int64_t>(tmpImpropers.size()), 5}, options_int);
	auto improp_acc = impropers_tensor.accessor<int64_t, 2>();

	torch::Tensor pairs_tensor = torch::empty({static_cast<int64_t>(tmpPairs.size()), 2}, options_int);
	auto pairs_acc = pairs_tensor.accessor<int64_t, 2>();

	torch::Tensor tetras_tensor = torch::empty({static_cast<int64_t>(tmpTetras.size()), 5}, options_int);
	auto tetras_acc = tetras_tensor.accessor<int64_t, 2>();

	torch::Tensor cistrans_tensor = torch::empty({static_cast<int64_t>(tmpCisTrans.size()), 5}, options_int);
	auto cistrans_acc = cistrans_tensor.accessor<int64_t, 2>();

	torch::Tensor encoding_tensor = torch::empty({static_cast<int64_t>(tmpEncoding.size()), static_cast<int64_t>(tmpEncoding[0].size())}, options_float);
	auto enc_acc = encoding_tensor.accessor<float, 2>();

	
	//Copy data to the tensors
	// atoms
	for (int i = 0; i < numGhostParticles; i++) {
	  at_acc[i] = tmpAtomTypes[i];
	}

	// bonds
	for (int i = 0; i < n_edges; i++) {
	  bonds_acc[i][0] = tmpEdgeIdxs[i][0];
	  bonds_acc[i][1] = tmpEdgeIdxs[i][1];
	}

	// angles
	for (int i = 0; i < tmpAngles.size(); i++) {
		assert(tmpAngles[i].size() == 4);
		for (int j = 0; j < 4; j++) {
			angles_acc[i][j] = tmpAngles[i][j];
		}
	}	

	// propers
	for (int i = 0; i < tmpPropers.size(); i++) {
		assert(tmpPropers[i].size() == 5);
		for (int j = 0; j < 5; j++) {
			prop_acc[i][j] = tmpPropers[i][j];
		}
	}
	
	// impropers
	for (int i = 0; i < tmpImpropers.size(); i++) {
		assert(tmpImpropers[i].size() == 5);
		for (int j = 0; j < 5; j++) {
			improp_acc[i][j] = tmpImpropers[i][j];
		}
	}

	// pairs
	for (int i = 0; i < tmpPairs.size(); i++) {
		assert(tmpPairs[i].size() == 2);
		for (int j = 0; j < 2; j++) {
			pairs_acc[i][j] = tmpPairs[i][j];
		}
	}

	// tetras
	for (int i = 0; i < tmpTetras.size(); i++) {
		assert(tmpTetras[i].size() == 5);
		for (int j = 0; j < 5; j++) {
			tetras_acc[i][j] = tmpTetras[i][j];
		}
	}

	// cistrans
	for (int i = 0; i < tmpCisTrans.size(); i++) {
		assert(tmpCisTrans[i].size() == 5);
		for (int j = 0; j < 5; j++) {
			cistrans_acc[i][j] = tmpCisTrans[i][j];
		}
	}

	// encoding
	for (int i = 0; i < tmpEncoding.size(); i++) {
		for (int j = 0; j < tmpEncoding[i].size(); j++) {
			enc_acc[i][j] = tmpEncoding[i][j];
		}
	}

	//                         |---------------------------- fixedInputs ---------------------------------|
	// nnInputs = {pos, sigma, atoms, bonds, angles, propers, impropers, pairs, tetras, cistrans, encoding}
	fixedInputs = {atoms_tensor, bonds_tensor, angles_tensor, propers_tensor, impropers_tensor, pairs_tensor, tetras_tensor, cistrans_tensor, encoding_tensor};

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
double CudaCalcPyTorchForceE2EDiffConfKernel::execute(ContextImpl& context,bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    int numParticles = cu.getNumAtoms();
	int numGhostParticles = particleIndices.size();

	double sigma = context.getParameter("diff_sigma");
	torch::Tensor sigmaTensor = torch::ones({1}, options_float) * sigma;


    vector<Vec3> MDPositions;
    context.getPositions(MDPositions);

	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 1, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 3>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
		positions[i][0][0] = MDPositions[particleIndices[i]][0] *10;
		positions[i][0][1] = MDPositions[particleIndices[i]][1] *10;
		positions[i][0][2] = MDPositions[particleIndices[i]][2] *10;
	}

	vector<torch::jit::IValue> nnInputs = {};
	
	nnInputs.push_back(positionsTensor);
	nnInputs.push_back(sigmaTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");

	torch::Tensor noise = scale*nnModule.forward(nnInputs).toTensor();

	//std::cout << "tim_idx, sigfac, mean noise:" << tim_idx << " " << sigfac << " " << torch::pow(noise, 2).mean() << "\n";
	// get forces on positions as before
	if (includeForces) {

	  auto NNForce = noise.accessor<float, 2>();
	  
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

	return 0.0; // E2EDiffConf only updates forces, there is no energy

}
