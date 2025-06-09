%module mlforce

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

%include <std_string.i>
%include <std_vector.i>
%{
#include "PyTorchForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}
namespace std {
  %template(vectori) vector<int>;
  %template(vectord) vector<double>;
  %template(vectordd) vector< vector<double> >;
  %template(vectorii) vector< vector<int> >;
  };

namespace PyTorchPlugin {

class PyTorchForce : public OpenMM::Force {
public:
	PyTorchForce(const std::string& file,
				 std::vector<std::vector<double> > targetFeatures,
				 const std::vector<int> particleIndices,
				 const std::vector<double> signalForceWeights,
				 double scale,
				 int assignFreq,
				 std::vector<std::vector<int> > restraintIndices,
				 const std::vector<double> restraintDistances,
				 double rmaxDelta,
				 double restraintK,
				 const std::vector<int> initialAssignment
				 );

	const std::string& getFile() const;
	const double getScale() const;
	const int getAssignFreq() const;
	const std::vector<std::vector<double> > getTargetFeatures() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	void setUsesPeriodicBoundaryConditions(bool periodic);
	bool usesPeriodicBoundaryConditions() const;
	int getNumGlobalParameters() const;
	int addGlobalParameter(const std::string& name, double defaultValue);
	const std::string& getGlobalParameterName(int index) const;
	void setGlobalParameterName(int index, const std::string& name);
	double getGlobalParameterDefaultValue(int index) const;
	void setGlobalParameterDefaultValue(int index, double defaultValue);
	const std::vector<std::vector<int> > getRestraintIndices() const;
	const std::vector<double> getRestraintDistances() const;
	const std::vector<double> getRestraintParams() const;
	const std::vector<int> getInitialAssignment() const;
};

class PyTorchForceE2E : public OpenMM::Force {
public:
	PyTorchForceE2E(const std::string& file,
					const std::vector<int> particleIndices,
					const std::vector<double> signalForceWeights,
					double scale,
					double offset);

	const std::string& getFile() const;
	const double getScale() const;
	const double getOffset() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	void setUsesPeriodicBoundaryConditions(bool periodic);
	bool usesPeriodicBoundaryConditions() const;
	int getNumGlobalParameters() const;
	int addGlobalParameter(const std::string& name, double defaultValue);
	const std::string& getGlobalParameterName(int index) const;
	void setGlobalParameterName(int index, const std::string& name);
	double getGlobalParameterDefaultValue(int index) const;
	void setGlobalParameterDefaultValue(int index, double defaultValue);
};

class PyTorchForceE2EDirect : public OpenMM::Force {
public:
	PyTorchForceE2EDirect(const std::string& file,
						  const std::vector<int> particleIndices,
						  const std::vector<double> signalForceWeights,
						  double scale,
						  const std::vector<int> atomTypes,
						  const std::vector<std::vector<int>> edgeIndices,
						  const std::vector<int> edgeTypes,
						  bool useAttr);

	const std::string& getFile() const;
	const double getScale() const;
	const std::vector<int> getAtomTypes() const;
	const std::vector<std::vector<int>> getEdgeIndices() const;
	const std::vector<int> getEdgeTypes() const;
	const bool getUseAttr() const;
	const std::vector<int> getParticleIndices() const;
	const std::vector<double> getSignalForceWeights() const;
	void setUsesPeriodicBoundaryConditions(bool periodic);
	bool usesPeriodicBoundaryConditions() const;
	int getNumGlobalParameters() const;
	int addGlobalParameter(const std::string& name, double defaultValue);
	const std::string& getGlobalParameterName(int index) const;
	void setGlobalParameterName(int index, const std::string& name);
	double getGlobalParameterDefaultValue(int index) const;
	void setGlobalParameterDefaultValue(int index, double defaultValue);
};

class PyTorchForceE2EDiffConf : public OpenMM::Force {
	public:
		PyTorchForceE2EDiffConf(const std::string& file,
							  const std::vector<int> particleIndices,
							  const std::vector<double> signalForceWeights,
							  double scale,
							  const std::vector<int> atoms,
							  const std::vector<std::vector<int>> bonds,
							  const std::vector<std::vector<int>> angles,
							  const std::vector<std::vector<int>> propers,
							  const std::vector<std::vector<int>> impropers,
							  const std::vector<std::vector<int>> pairs,
							  const std::vector<std::vector<int>> tetras,
							  const std::vector<std::vector<int>> cistrans,
							  const std::vector<std::vector<float>> encoding
							);
	
		const std::string& getFile() const;
		const double getScale() const;
		const std::vector<int> getAtomTypes() const;
		const std::vector<std::vector<int>> getEdgeIndices() const;
		const std::vector<std::vector<int>> getAngles() const;
		const std::vector<std::vector<int>> getPropers() const;
		const std::vector<std::vector<int>> getImpropers() const;
		const std::vector<std::vector<int>> getPairs() const;
		const std::vector<std::vector<int>> getTetras() const;
		const std::vector<std::vector<int>> getCisTrans() const;
		const std::vector<std::vector<float>> getEncoding() const;
		
		const std::vector<int> getParticleIndices() const;
		const std::vector<double> getSignalForceWeights() const;
		void setUsesPeriodicBoundaryConditions(bool periodic);
		bool usesPeriodicBoundaryConditions() const;
		int getNumGlobalParameters() const;
		int addGlobalParameter(const std::string& name, double defaultValue);
		const std::string& getGlobalParameterName(int index) const;
		void setGlobalParameterName(int index, const std::string& name);
		double getGlobalParameterDefaultValue(int index) const;
		void setGlobalParameterDefaultValue(int index, double defaultValue);
	};
 
}
