#include "ReferencePyTorchKernels.h"
#include "PyTorchForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>
#include <assert.h>

using namespace PyTorchPlugin;
using namespace OpenMM;
using namespace std;

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractPositions(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->positions);
}

/**
 * @brief
 *
 * @param context
 * @return vector<Vec3>&
 */
static vector<Vec3>& extractForces(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((vector<Vec3>*) data->forces);
}
/**
 * @brief
 *
 * @param context
 * @return Vec3*
 */
static Vec3* extractBoxVectors(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return (Vec3*) data->periodicBoxVectors;
}

/**
 * @brief
 *
 * @param context
 * @return map<string, double>&
 */
static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
	ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
	return *((map<string, double>*) data->energyParameterDerivatives);
}


/**
 * @brief
 *
 * @param context
 * @param numParticles
 * @return std::vector<double>
 */
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

// -------------------- Least-squares rigid-rotation remover (memory-efficient) --------------------
// Returns corrected_dX (N,3)
torch::Tensor remove_rigid_rotation_lstsq_loop(
	const torch::Tensor& coords_in,
    const torch::Tensor& dX_in,
    bool center = true,
    double lambda_reg = 1e-8
) {
    TORCH_CHECK(coords_in.dim() == 2 && coords_in.size(1) == 3, "coords must be (N,3)");
    TORCH_CHECK(dX_in.dim() == 2 && dX_in.size(1) == 3, "dX must be (N,3)");
    TORCH_CHECK(coords_in.size(0) == dX_in.size(0), "coords and dX must have same N");

    auto device = coords_in.device();
    auto dtype  = coords_in.dtype();
    const int64_t N = coords_in.size(0);

    // center if requested
    auto coords = coords_in;
    auto dX     = dX_in;
    if (center) {
	    auto coords_mean = coords_in.mean(0, /*keepdim=*/true);
        auto dX_mean     = dX_in.mean(0, /*keepdim=*/true);
        coords = coords_in - coords_mean;
        dX     = dX_in - dX_mean;
    }

    // Prepare accumulators on device/dtype
    auto I3 = torch::eye(3, torch::TensorOptions().device(device).dtype(dtype));
    auto A = torch::zeros({3,3}, torch::TensorOptions().device(device).dtype(dtype));
    auto b = torch::zeros({3},   torch::TensorOptions().device(device).dtype(dtype));

    // Loop accumulate A and b in a memory-friendly way (vectorized in chunks if desired)
    const int64_t chunk = 1 << 16; // process in chunks to reduce kernel launches if N large
    for (int64_t start = 0; start < N; start += chunk) {
        int64_t end = std::min<int64_t>(N, start + chunk);
        auto r_chunk = coords.slice(0, start, end); // (M,3)
        auto d_chunk = dX.slice(0, start, end);     // (M,3)

        // rsq: (M,)
        auto rsq = torch::sum(r_chunk * r_chunk, 1); 

        // Compute per-chunk A contribution: sum ( rsq_i * I - r_i r_i^T )
        // Use vectorized outer: (M,3,1) x (M,1,3) => (M,3,3)
        auto r_col = r_chunk.unsqueeze(2);
        auto r_row = r_chunk.unsqueeze(1);
        auto outer = r_col.matmul(r_row); // (M,3,3)

        // rsq * I per sample
        auto rsq_exp = rsq.view({-1,1,1}); // (M,1,1)
        auto rsqI = rsq_exp * I3.view({1,3,3}); // (M,3,3)

        auto A_per = rsqI - outer; // (M,3,3)
        auto A_sum = torch::sum(A_per, 0); // (3,3)
        A += A_sum;

        // b contribution: sum r_i x d_i
        auto cross_rd = torch::cross(r_chunk, d_chunk, /*dim=*/1); // (M,3)
        auto b_sum = torch::sum(cross_rd, 0); // (3,)
        b += b_sum;
    }

    // Regularize
    if (lambda_reg > 0.0) {
        A = A + lambda_reg * I3;
    }

    // Solve A w = b
    auto b_col = b.view({3,1});
    torch::Tensor w_col;
    // Try available linalg::solve signatures robustly
	w_col = torch::linalg::solve(A, b_col, /*left=*/true);
    auto w = w_col.view({3});

    // Compute corrected displacements: dX_corr = dX - w x r
    auto w_expand = w.view({1,3}).expand({coords.size(0),3});
    auto wx = torch::cross(w_expand, coords, /*dim=*/1);
    auto dX_corr = dX - wx;

    return dX_corr;
}


ReferenceCalcPyTorchForceE2EDiffConfKernel::~ReferenceCalcPyTorchForceE2EDiffConfKernel() {
}


/**
 * @brief
 *
 * @param system
 * @param force
 * @param nnModule
 */
void ReferenceCalcPyTorchForceE2EDiffConfKernel::initialize(const System& system, const PyTorchForceE2EDiffConf& force, torch::jit::script::Module& nnModule) {
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
	vector<vector<double>> tmpEncoding = force.getEncoding();
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
		  enc_acc[i][j] = float(tmpEncoding[i][j]);
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

}

/**
 * @brief
 *
 * @param context
 * @param includeForces
 * @param includeEnergy
 * @return double
 */
double ReferenceCalcPyTorchForceE2EDiffConfKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

  // Get the  positions from the context (previous step)
	vector<Vec3>& MDPositions = extractPositions(context);
	vector<Vec3>& MDForce = extractForces(context);

	double sigma = context.getParameter("diff_sigma");
	torch::Tensor sigmaTensor = torch::ones({1}, options_float) * sigma;

	int numGhostParticles = particleIndices.size();
	
	torch::Tensor positionsTensor = torch::empty({numGhostParticles, 1, 3}, options_float.requires_grad(true));

	auto positions = positionsTensor.accessor<float, 3>();
	//Copy positions to the tensor
	for (int i = 0; i < numGhostParticles; i++) {
	  positions[i][0][0] = MDPositions[particleIndices[i]][0]*10;  // convert positions to angstroms
	  positions[i][0][1] = MDPositions[particleIndices[i]][1]*10;
	  positions[i][0][2] = MDPositions[particleIndices[i]][2]*10;
	}
	
	vector<torch::jit::IValue> nnInputs = {};

	nnInputs.push_back(positionsTensor);
	nnInputs.push_back(sigmaTensor);
	for ( auto &ten : fixedInputs ) {
	  nnInputs.push_back(ten);
	}

	//std::cout << "get_diffusion_noise device:" << get_diffusion_noise.device();
	torch::Tensor noise = scale*nnModule.forward(nnInputs).toTensor();
	noise = remove_rigid_rotation_lstsq_loop(positionsTensor.squeeze(1), noise);
	
	// get forces on positions as before
	if (includeForces) {

	  auto NNForce = noise.accessor<float, 2>();
		
	  for (int i = 0; i < numGhostParticles; i++) {
		MDForce[particleIndices[i]][0] += double(NNForce[i][0]);
		MDForce[particleIndices[i]][1] += double(NNForce[i][1]);
		MDForce[particleIndices[i]][2] += double(NNForce[i][2]);
	  }

	}
	return 0.0; // E2EDiffConf only updates forces, there is no energy
  }
