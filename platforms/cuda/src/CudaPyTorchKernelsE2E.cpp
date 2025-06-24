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

CudaCalcPyTorchForceE2EKernel::CudaCalcPyTorchForceE2EKernel(string name, const Platform& platform, CudaContext& cu): CalcPyTorchForceE2EKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

/**
 * @brief Destroy the Cuda CalcPy Torch Force Kernel:: Cuda CalcPy Torch Force Kernel object
 *
 */
CudaCalcPyTorchForceE2EKernel::~CudaCalcPyTorchForceE2EKernel() {
  cuDevicePrimaryCtxRelease(cu.getDevice());
}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */

void CudaCalcPyTorchForceE2EKernel::initialize(const System& system, const PyTorchForceE2E& force, torch::jit::script::Module& nnModule) {

	this->nnModule = nnModule; 
	nnModule.to(torch::kCPU);
	nnModule.eval();

	scale = force.getScale();
	offset = force.getOffset();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();

	usePeriodic = force.usesPeriodicBoundaryConditions();
	useLambda = force.usesLambda();
	
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
    torch::TensorOptions options_gpu = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	// Pop the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

	torch::TensorOptions options_cpu = torch::TensorOptions().
	  device(torch::kCPU).
	  dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
	
	// prepare all-to-all edge tensors
    std::vector<vector<int64_t>> edges;
    for (int64_t i = 0; i < numGhostParticles; i++) {
      for (int64_t j = 0; j < numGhostParticles; j++) {
        if (i != j) {
          vector<int64_t> p = {i,j};
          edges.push_back(p);
        }
      }
    }
    int num_edges = edges.size();
	
    edge_idxs = torch::empty({2, static_cast<int64_t>(num_edges)},
                             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto edge_acc = edge_idxs.accessor<int64_t, 2>();

    //Copy indices to the tensor
    for (int i = 0; i < num_edges; i++) {
      edge_acc[0][i] = edges[i][0];
      edge_acc[1][i] = edges[i][1];
    }

	edge_attrs = torch::zeros({num_edges, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    batch = torch::zeros({numGhostParticles}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

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
double CudaCalcPyTorchForceE2EKernel::execute(ContextImpl& context,bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
    int numParticles = cu.getNumAtoms();
	int numGhostParticles = particleIndices.size();

    vector<Vec3> MDPositions;
    context.getPositions(MDPositions);
    torch::Tensor positionsTensor = torch::empty({static_cast<int64_t>(numGhostParticles), 3},
        cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    if (cu.getUseDoublePrecision()) {
        auto positions = positionsTensor.accessor<double, 2>();
        for (int i = 0; i < numGhostParticles; i++) {
            positions[i][0] = MDPositions[particleIndices[i]][0];
            positions[i][1] = MDPositions[particleIndices[i]][1];
            positions[i][2] = MDPositions[particleIndices[i]][2];
        }
    }
    else {
        auto positions = positionsTensor.accessor<float, 2>();
        for (int i = 0; i < numGhostParticles; i++) {
            positions[i][0] = MDPositions[particleIndices[i]][0];
            positions[i][1] = MDPositions[particleIndices[i]][1];
            positions[i][2] = MDPositions[particleIndices[i]][2];
        }
    }

	int n_signals;
	if (useLambda) {
	  n_signals = 4;
	} else {
	  n_signals = 3;
	}
	
	torch::Tensor signalsTensor = torch::empty({numGhostParticles, n_signals}, torch::kFloat64);
    std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);

	auto signals = signalsTensor.accessor<double, 2>();

	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  for (int j = 0; j < n_signals; j++) {
		signals[i][j] = globalVariables[4*i + j];
	  }
	}

    torch::TensorOptions options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);

    signalsTensor = signalsTensor.to(options);
    positionsTensor = positionsTensor.to(options);
    positionsTensor.requires_grad_(true);
    signalsTensor.requires_grad_(true) ;
	
	// Run the pytorch model and get the energy
	vector<torch::jit::IValue> nnInputs = {signalsTensor, positionsTensor, edge_idxs, edge_attrs, batch};

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");

    // outputTensor : energy
    torch::Tensor energyTensor = scale*nnModule.forward(nnInputs).toTensor() + offset;

	// get forces on positions as before
	if (includeForces) {
	    energyTensor.backward();

		// check if positions have gradients
		auto forceTensor = torch::zeros_like(positionsTensor);
		auto signalDerivTensor = torch::zeros_like(signalsTensor);

		/*The .clone() function is used to create a new tensor with the same values as positionsTensor.grad() 
		to ensure that it is not affected by subsequent operations.*/
		forceTensor = - positionsTensor.grad().clone(); 
		signalDerivTensor = signalsTensor.grad().clone(); 

		positionsTensor.grad().zero_(); // clear the gradients before the next round of backpropagation or gradient computation.
		signalsTensor.grad().zero_();

		map<string, double> &energyParamDerivs = cu.getEnergyParamDerivWorkspace();

		// saving signals derivatives to context
		if (cu.getUseDoublePrecision()) {
		  if (!(forceTensor.dtype() == torch::kFloat64))
			forceTensor = forceTensor.to(torch::kFloat64);

		  if (!(signalDerivTensor.dtype() == torch::kFloat64))
            signalDerivTensor = signalDerivTensor.to(torch::kFloat64);

		  auto signalDerivData = signalDerivTensor.accessor<double, 2>();
			for (int i = 0; i < numGhostParticles; i++) {
				for (int j=0; j < n_signals; j++){
					energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += signalDerivData[i][j]*signalForceWeights[j];
				}
			}

		} else {
		  if (!(forceTensor.dtype() == torch::kFloat32))
			forceTensor = forceTensor.to(torch::kFloat32);

		  if (!(signalDerivTensor.dtype() == torch::kFloat32))
            signalDerivTensor = signalDerivTensor.to(torch::kFloat32);

		  auto signalDerivData = signalDerivTensor.accessor<float, 2>();
		  for (int i = 0; i < numGhostParticles; i++) {
			for (int j=0; j < n_signals; j++) {
			  energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += signalDerivData[i][j]*signalForceWeights[j];
			}
		  }		
		}

		// sending atomic forces to cuda context
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
	const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy;
}
