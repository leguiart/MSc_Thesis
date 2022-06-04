#include "VX3_MaterialLink.h"
#include "VX3_VoxelyzeKernel.cuh"

VX3_MaterialLink::VX3_MaterialLink(CVX_MaterialLink* p, VX3_VoxelyzeKernel* k):
VX3_MaterialVoxel((CVX_MaterialVoxel *)p, k),
_a1(p->_a1), _a2(p->_a2), _b1(p->_b1), _b2(p->_b2), _b3(p->_b3),
_sqA1(p->_sqA1), _sqA2xIp(p->_sqA2xIp), _sqB1(p->_sqB1), 
_sqB2xFMp(p->_sqB2xFMp), _sqB3xIp(p->_sqB3xIp) {
	for (int i=0;i<k->h_voxelMats.size();i++) {
		if (k->h_voxelMats[i] == p->vox1Mat) {
			vox1Mat = &k->d_voxelMats[i];
		} //vox1Mat and vox2Mat could be the same Mat
		if (k->h_voxelMats[i] == p->vox2Mat) {
			vox2Mat = &k->d_voxelMats[i];
		}
	}
	if (vox1Mat==NULL || vox2Mat==NULL) {
		printf(COLORCODE_BOLD_RED "ERROR: Cannot find voxel for voxelMats.\n");
	}
}
VX3_MaterialLink::~VX3_MaterialLink() {
}

__device__ VX3_MaterialLink::VX3_MaterialLink(VX3_MaterialVoxel* mat1, VX3_MaterialVoxel* mat2)
{
	vox1Mat = mat1;
	vox2Mat = mat2;

	updateAll();

}

__device__ VX3_MaterialLink& VX3_MaterialLink::operator=(const VX3_MaterialLink& vIn)
{
	VX3_MaterialVoxel::operator=(vIn); //set base VX3_MaterialVoxel class variables equal

	vox1Mat = vIn.vox1Mat;
	vox2Mat = vIn.vox2Mat;
	_a1 = vIn._a1;
	_a2 = vIn._a2;
	_b1 = vIn._b1;
	_b2 = vIn._b2;
	_b3 = vIn._b3;
	_sqA1 = vIn._sqA1;
	_sqA2xIp = vIn._sqA2xIp;
	_sqB1 = vIn._sqB1;
	_sqB2xFMp = vIn._sqB2xFMp;
	_sqB3xIp = vIn._sqB3xIp;

	return *this;
}

__device__ bool VX3_MaterialLink::updateAll()
{
	nomSize = 0.5*(vox1Mat->nomSize + vox2Mat->nomSize); //these should be the same...

	r=(int)(0.5*(vox1Mat->r + vox2Mat->r));
	g=(int)(0.5*(vox1Mat->g + vox2Mat->g));
	b=(int)(0.5*(vox1Mat->b + vox2Mat->b));
	a=(int)(0.5*(vox1Mat->a + vox2Mat->a));

	rho = 0.5f*(vox1Mat->rho + vox2Mat->rho);
	alphaCTE = 0.5f*(vox1Mat->alphaCTE + vox2Mat->alphaCTE);
	muStatic = 0.5f*(vox1Mat->muStatic + vox2Mat->muStatic);
	muKinetic = 0.5f*(vox1Mat->muKinetic + vox2Mat->muKinetic);
	zetaInternal = 0.5f*(vox1Mat->zetaInternal + vox2Mat->zetaInternal);
	zetaGlobal = 0.5f*(vox1Mat->zetaGlobal + vox2Mat->zetaGlobal);
	zetaCollision= 0.5f*(vox1Mat->zetaCollision + vox2Mat->zetaCollision);

	extScale=VX3_Vec3D<>(1.0, 1.0, 1.0);

	//failure stress (f) is the minimum of the two failure stresses, or if both are -1.0f it should also be -1.0f to denote no failure specified
	float stressFail=-1.0f, /*strainFail=-1.0f,*/ f1=vox1Mat->sigmaFail, f2=vox2Mat->sigmaFail;
	if (f1 == -1.0f) stressFail = f2; //-1.0f or vox2Mat fail
	else if (f2 == -1.0f) stressFail = f1; //vox1Mat fail
	else stressFail = f1 < f2 ? f1 : f2; //the lesser stress denotes failure

	if (vox1Mat->linear && vox2Mat->linear) setModelLinear(2.0f*vox1Mat->E*vox2Mat->E/(vox1Mat->E+vox2Mat->E), stressFail);
	else { //at least 1 bilinear or data-based, so build up data points and apply it.
		VX3_dVector<float> newStressValues, newStrainValues;
		newStressValues.push_back(0.0f);
		newStrainValues.push_back(0.0f);

		//step up through ascending strains data points (could alternate randomly between vox1Mat and vox2Mat points
		int dataIt1 = 1, dataIt2 = 1; //iterators through each data point of the model
		while (dataIt1 < (int)vox1Mat->d_strainData.size() && dataIt2 < (int)vox2Mat->d_strainData.size()){
			float strain = FLT_MAX; //strain for the next data point is the smaller of the two possible next strain points (but we have to make sure we don't access off the end of one of the arrays)
			if (dataIt1 < (int)vox1Mat->d_strainData.size()) strain = vox1Mat->d_strainData[dataIt1];
			if (dataIt2 < (int)vox2Mat->d_strainData.size() && vox2Mat->d_strainData[dataIt2]<strain) strain = vox2Mat->d_strainData[dataIt2];
			else assert(strain != FLT_MAX); //this should never happen

			if (strain == vox1Mat->d_strainData[dataIt1]) dataIt1++;
			if (strain == vox2Mat->d_strainData[dataIt2]) dataIt2++;


			float modulus1 = vox1Mat->modulus(strain-FLT_EPSILON);
			float modulus2 = vox2Mat->modulus(strain-FLT_EPSILON);
			float thisModulus = 2.0f*modulus1*modulus2/(modulus1+modulus2);

			//add to the new strain/stress values
			int lastDataIndex = newStrainValues.size()-1;

			newStrainValues.push_back(strain);
			newStressValues.push_back(newStressValues[lastDataIndex] + thisModulus*(strain - newStrainValues[lastDataIndex])); //springs in series equation
		}

		setModel(newStrainValues.size(), &newStrainValues[0], &newStressValues[0]);

		//override failure points in case no failure was specified before (as possible in combos of linear and bilinear materials)
		//yield point is handled correctly in setModel.
		sigmaFail = stressFail;
		epsilonFail = stressFail==-1.0f ? -1.0f : strain(stressFail);
	}

	//poissons ratio: choose such that Ehat ends up according to spring in series of Ehat1 and EHat2
	if (vox1Mat->nu==0 && vox2Mat->nu==0) nu = 0;
	else { //poissons ratio: choose such that Ehat ends up according to spring in series of Ehat1 and EHat2
		float tmpEHat = 2*vox1Mat->_eHat*vox2Mat->_eHat/(vox1Mat->_eHat+vox2Mat->_eHat);
		float tmpE = youngsModulus();
		//completing the square algorithm to solve for nu.
		//eHat = E/((1-2nu)(1+nu)) -> E/EHat = -2nu^2-nu+1 -> nu^2+0.5nu = (EHat+E)/(2EHat)
		float c2 = (tmpEHat-tmpE)/(2*tmpEHat)+0.0625; //nu^2+0.5nu+0.0625 = c2 -> (nu+0.25)^2 = c2
		nu = sqrt(c2)-0.25; //from solving above
	}

	return updateDerived();
}

__device__ bool VX3_MaterialLink::updateDerived() 
{
	VX3_MaterialVoxel::updateDerived(); //update base VX3_Material class derived variables

	//stiffnesses terms for links
	float L = (float)nomSize;
	_a1 = E*L; //EA/L : Units of N/m
	_a2 = E * L*L*L / (12.0f*(1+nu)); //GJ/L : Units of N-m
	_b1 = E*L; //12EI/L^3 : Units of N/m
	_b2 = E*L*L/2.0f; //6EI/L^2 : Units of N (or N-m/m: torque related to linear distance)
	_b3 = E*L*L*L/6.0f; //2EI/L : Units of N-m
	
	//damping sqrt(mk) terms (with sqrt(m) factored out)
	_sqA1=sqrt(_a1);
	_sqA2xIp=sqrt(_a2*L*L/6.0f);
	_sqB1=sqrt(_b1);
	_sqB2xFMp=sqrt(_b2*L/2.0f);
	_sqB3xIp=sqrt(_b3*L*L/6.0f);

	return true;
}
