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
				 double restraintK
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
};

}
