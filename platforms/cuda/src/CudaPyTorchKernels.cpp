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

/**
 * @brief
 *
 * @param ptr
 * @param nRows
 * @param nCols
 * @return std::vector<std::vector<double>>
 */

 // this function converts a 1D tensor into a 2D vector (distMat). Each row of the input array is mapped to a row in the resulting 2D vector. 
 // distMat has nRows rows and nCols columns.
static std::vector<std::vector<double> > tensorTo2DVec(double* ptr, int nRows, int nCols) {
	std::vector<std::vector<double> > distMat(nRows, std::vector<double>(nCols));
	for (int i=0; i<nRows; i++) {
		std::vector<double> vec(ptr+nCols*i, ptr+nRows*(i+1));
		distMat[i] = vec;
	}
	return distMat;
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

static std::vector<int> getReverseAssignment(std::vector<int> assignment) {
	int n = assignment.size();
	std::vector<int> rev_assignment(n, -1);
	for (int i=0; i<n; i++) {
		if ((assignment[i] >= 0) && (assignment[i] < n)) {
		  rev_assignment[assignment[i]] = i;
		}
	}
	return rev_assignment;
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

CudaCalcPyTorchForceKernel::CudaCalcPyTorchForceKernel(string name, const Platform& platform, CudaContext& cu) : CalcPyTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcPyTorchForceKernel::~CudaCalcPyTorchForceKernel() {
    cuDevicePrimaryCtxRelease(cu.getDevice());
}

/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */

void CudaCalcPyTorchForceKernel::initialize(const System& system, const PyTorchForce& force, torch::jit::script::Module& nnModule) {

	this->nnModule = nnModule;
	nnModule.to(torch::kCPU);
	nnModule.eval();

	int numParticles = system.getNumParticles();

	// get info from PyTorchForce object
	
	scale = force.getScale();
	assignFreq = force.getAssignFreq();
	particleIndices = force.getParticleIndices();
	signalForceWeights = force.getSignalForceWeights();
	targetRestraintIndices = force.getRestraintIndices();
	targetRestraintDistances = force.getRestraintDistances();
	targetRestraintParams = force.getRestraintParams();
	rmax_delta = targetRestraintParams[0];
	restraint_k = targetRestraintParams[1];

	assignment = force.getInitialAssignment();
	reverse_assignment = getReverseAssignment(assignment);
	
	numRestraints = targetRestraintDistances.size();
	for (int i = 0; i < numRestraints; i++) {
		r0sq.push_back(targetRestraintDistances[i]*targetRestraintDistances[i]);
		rmax.push_back(targetRestraintDistances[i] + rmax_delta);
		restraint_b.push_back(0.5*restraint_k*(r0sq[i] - rmax[i]*rmax[i]));
	}

	usePeriodic = force.usesPeriodicBoundaryConditions();
	int numGhostParticles = particleIndices.size();

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
	
	//get target features
	targetFeatures = force.getTargetFeatures();
	targetFeaturesTensor = torch::zeros({static_cast<int64_t>(targetFeatures.size()),
		static_cast<int64_t>(targetFeatures[0].size())},
		torch::kFloat64);

	for (std::size_t i = 0; i < targetFeatures.size(); i++) // num_rows (num_gp)
		targetFeaturesTensor.slice(0, i, i+1) = torch::from_blob(targetFeatures[i].data(), 
			{static_cast<int64_t>(targetFeatures[0].size())}, // num_columns (num_features)
			torch::TensorOptions().dtype(torch::kFloat64));

	signalFW_tensor = torch::from_blob(signalForceWeights.data(),
					{static_cast<int64_t>(signalForceWeights.size())},
					torch::kFloat64);

	torch::TensorOptions options_cpu = torch::TensorOptions().
		device(torch::kCPU).
		dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	targetFeaturesTensor = targetFeaturesTensor.to(options_cpu);
	signalFW_tensor= signalFW_tensor.to(options_cpu);
	restraintForceTensor = torch::zeros({numParticles, 3}, options_cpu);
	paddedForceTensor = torch::zeros({numParticles, 3}, options_cpu);

	
	if (usePeriodic) {
		boxVectorsTensor = torch::empty({3, 3}, options_cpu);
	}

	// Inititalize CUDA objects for OpenMM context

	ContextSelector selector(cu); // Switch to the OpenMM context
	map<string, string> defines;
	CUmodule program = cu.createModule(CudaPyTorchKernelSources::PyTorchForce, defines);
	copyInputsKernel = cu.getKernel(program, "copyInputs");
	addForcesKernel = cu.getKernel(program, "addForces");

	step_count = 0;
}


/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double CudaCalcPyTorchForceKernel::execute(ContextImpl& context,bool includeForces, bool includeEnergy) {
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
	
	torch::Tensor signalsTensor = torch::empty({numGhostParticles, 4}, torch::kFloat64);
	std::vector<double> globalVariables = extractContextVariables(context, numGhostParticles);
	signalsTensor = torch::from_blob(globalVariables.data(),
		{static_cast<int64_t>(numGhostParticles), 4}, torch::kFloat64);
	torch::TensorOptions options = torch::TensorOptions().device(torch::kCPU)
		.dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);

	signalsTensor = signalsTensor.to(options);
	positionsTensor = positionsTensor.to(options);
	positionsTensor.requires_grad_(true);
	signalsTensor.requires_grad_(true) ;
	// Run the pytorch model and get the energy
	auto charges = signalsTensor.index({Slice(), 0}); 
	vector<torch::jit::IValue> nnInputs = {positionsTensor, charges};

	// Copy the box vector
	if (usePeriodic) {
	  Vec3 box[3];
	  cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
	  boxVectorsTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);

	  boxVectorsTensor = boxVectorsTensor.to(options);
	  nnInputs.push_back(boxVectorsTensor);
	}

	// synchronizing the current context before switching to PyTorch
	CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
	
	// outputTensor is the gp features(AEV)
	torch::Tensor outputTensor = nnModule.forward(nnInputs).toTensor();
	int outputTensorSize = outputTensor.size(1);

	torch::Tensor AEVForceWeights = torch::ones({outputTensorSize}, options); 
	torch::Tensor allForceWeights = torch::cat({ AEVForceWeights, signalFW_tensor}, 0); // all force weights (features + attributes)
	
	torch::Tensor ghFeaturesTensor = torch::cat({outputTensor, signalsTensor}, 1);
	allForceWeights = allForceWeights.to(options); 
	ghFeaturesTensor = ghFeaturesTensor.to(options);
	
    // call Hungarian algorithm to determine mapping (and loss)
    if (assignFreq > 0) {
	  if (step_count % assignFreq == 0) {
		// torch:
		torch::Tensor distMatTensor = at::norm((ghFeaturesTensor.index({Slice(), None})
											   - targetFeaturesTensor) , 2, 2);

		if (!cu.getUseDoublePrecision())
		  distMatTensor=distMatTensor.to(torch::kFloat64);

		std::vector<std::vector<double>> distMatrix = tensorTo2DVec(distMatTensor.data_ptr<double>(),
																	numGhostParticles,
																	static_cast<int>(targetFeaturesTensor.size(0)));
		// call Hungarian algorithm to determine mapping (and loss)
		assignment = hungAlg.Solve(distMatrix);
		reverse_assignment = getReverseAssignment(assignment);
	  }
	}
	for (std::size_t i=0; i<assignment.size(); i++) {
	  context.setParameter("assignment_g"+std::to_string(i), assignment[i]);
	}
	step_count += 1;

	// reorder the targetFeaturesTensor using the mapping
	// and then creates a deep copy of the reorder tensor in the variable reFeaturesTensor. 
	torch::Tensor reFeaturesTensor = targetFeaturesTensor.index({{torch::tensor(assignment)}}).clone();

	torch::Tensor diff = ghFeaturesTensor - reFeaturesTensor;
	torch::Tensor diffSquared = diff * diff;
	torch::Tensor weightedDiff = diffSquared * allForceWeights;
	torch::Tensor sumWeightedDiff = weightedDiff.sum();
	torch::Tensor mse = sumWeightedDiff / (ghFeaturesTensor.size(0) * ghFeaturesTensor.size(1));
	torch::Tensor energyTensor = scale * mse.clone();
	
	// calculate force on the signals (first, clip out signals from the end of features)
	torch::Tensor targtSignalsTensor = reFeaturesTensor.narrow(1, -4, 4); // only target attributes
	double restraint_energy = 0;

	if (cu.getUseDoublePrecision()) {
	  auto rfaccessor = restraintForceTensor.accessor<double,2>();
	  // compute energies and forces from restraints
	  for (int i = 0; i < numRestraints; i++) {
		int g1idx = reverse_assignment[targetRestraintIndices[i][0]];
		int g2idx = reverse_assignment[targetRestraintIndices[i][1]];
		OpenMM::Vec3 r = MDPositions[g1idx] - MDPositions[g2idx];
		double rlensq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
		if (rlensq > r0sq[i]) {
		  double rlen = sqrt(rlensq);
		  if (rlen < rmax[i]) {
			double dr = rlen-targetRestraintDistances[i];
			restraint_energy += 0.5*scale*restraint_k*dr*dr;
			if (includeForces) {
			  OpenMM::Vec3 dvdx = scale*restraint_k*dr*r/rlen;
			  for (int j = 0; j < 3; j++) {
				rfaccessor[g1idx][j] -= dvdx[j];
				rfaccessor[g2idx][j] += dvdx[j];
			  }
			}
		  } else {
			restraint_energy += scale*(restraint_k*(rmax[i]-targetRestraintDistances[i])*rlen +	restraint_b[i]);
			if (includeForces) {
			  OpenMM::Vec3 dvdx = scale*restraint_k*(rmax[i]-targetRestraintDistances[i])*r/rlen;
			  for (int j = 0; j < 3; j++) {
				rfaccessor[g1idx][j] -= dvdx[j];
				rfaccessor[g2idx][j] += dvdx[j];
			  }
			}
		  }
		}
	  }
    } else {
	  auto rfaccessor = restraintForceTensor.accessor<float,2>();
	  // compute energies and forces from restraints
	  for (int i = 0; i < numRestraints; i++) {
		int g1idx = reverse_assignment[targetRestraintIndices[i][0]];
		int g2idx = reverse_assignment[targetRestraintIndices[i][1]];
		OpenMM::Vec3 r = MDPositions[g1idx] - MDPositions[g2idx];
		float rlensq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
		if (rlensq > r0sq[i]) {
		  float rlen = sqrt(rlensq);
		  if (rlen < rmax[i]) {
			float dr = rlen-targetRestraintDistances[i];
			restraint_energy += 0.5*scale*restraint_k*dr*dr;
			if (includeForces) {
			  OpenMM::Vec3 dvdx = scale*restraint_k*dr*r/rlen;
			  for (int j = 0; j < 3; j++) {
				rfaccessor[g1idx][j] -= dvdx[j];
				rfaccessor[g2idx][j] += dvdx[j];
			  }
			}
		  } else {
			restraint_energy += scale*(restraint_k*(rmax[i]-targetRestraintDistances[i])*rlen +	restraint_b[i]);
			if (includeForces) {
			  OpenMM::Vec3 dvdx = scale*restraint_k*(rmax[i]-targetRestraintDistances[i])*r/rlen;
			  for (int j = 0; j < 3; j++) {
				rfaccessor[g1idx][j] -= dvdx[j];
				rfaccessor[g2idx][j] += dvdx[j];
			  }
			}
		  }
		}
	  }
    }
	// get forces on positions as before
	if (includeForces) {

		/*The backward() function is then called on energyTensor to compute the gradients with respect to the tensors involved in the computation.
		For example outputTensor and reFeaturesTensor are involved in calculating the energyTensor. We can access of these tensors by:
		outputTensor.grad; // Gradients of energyTensor with respect to outputTensor
		reFeaturesTensor.grad; // Gradients of energyTensor with respect to reFeaturesTensor
		*/
		energyTensor.backward();
		// check if positions have gradients
		auto forceTensor = torch::zeros_like(positionsTensor);
		auto signalsGradTensor = torch::zeros_like(signalsTensor);
		/*The .clone() function is used to create a new tensor with the same values as positionsTensor.grad() 
		to ensure that it is not affected by subsequent operations.*/
		forceTensor = - positionsTensor.grad().clone();
		signalsGradTensor = signalsTensor.grad().clone(); 

		positionsTensor.grad().zero_(); // clear the gradients before the next round of backpropagation or gradient computation.
		signalsTensor.grad().zero_();

		map<string, double> &energyParamDerivs = cu.getEnergyParamDerivWorkspace();

		// saving signals derivatives to context
		//std::ofstream outputFile("/home/andre/code/mlforce_ft/data.txt" , std::ios::app);
		if (cu.getUseDoublePrecision()) {
			double parameter_deriv;
			auto signalsGradData = signalsGradTensor.accessor<double, 2>();

			for (int i = 0; i < numGhostParticles; i++) {
				for (int j=0; j<4; j++){
					
					energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += signalsGradData[i][j];
				}
			}

		} else {
			float parameter_deriv;
			auto signalsGradData = signalsGradTensor.accessor<float, 2>();

			for (int i = 0; i < numGhostParticles; i++) {
				for (int j=0; j<4; j++) {
					energyParamDerivs[PARAMETERNAMES[j]+std::to_string(i)] += signalsGradData[i][j];
				}
			}		
		}

		// sending atomic forces to cuda context
		paddedForceTensor = torch::zeros({numParticles, 3});
		paddedForceTensor.narrow(0,
			static_cast<int64_t>(particleIndices[0]),
			static_cast<int64_t>(particleIndices.size())).copy_(forceTensor);
		paddedForceTensor += restraintForceTensor; 
		
		//void* forceData = paddedForceTensor.data_ptr;
		const torch::Device device(torch::kCUDA, cu.getDeviceIndex());
		paddedForceTensor = paddedForceTensor.to(device);
		void* forceData = getTensorPointer(cu, paddedForceTensor);
		
		CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
		//cu.setAsCurrent();
		{
		  ContextSelector selector(cu); // Switch to the OpenMM context
		  int paddedNumAtoms = cu.getPaddedNumAtoms();
		  void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(),
			&cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
		  cu.executeKernel(addForcesKernel, forceArgs, numParticles);
		  CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
		}
	}
    const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy + restraint_energy;
}
