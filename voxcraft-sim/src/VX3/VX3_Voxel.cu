#include "VX3_Voxel.h"
#include <vector>

#include "VX3_External.h"
#include "VX3_Link.h"
#include "VX3_MaterialVoxel.h"
#include "VX3_MemoryCleaner.h"
#include "VX3_VoxelyzeKernel.cuh"

VX3_Voxel::VX3_Voxel(CVX_Voxel *p, VX3_VoxelyzeKernel *k)
    : matid(p->matid), ix(p->ix), iy(p->iy), iz(p->iz), pos(p->pos), linMom(p->linMom), orient(p->orient), angMom(p->angMom),
      boolStates(p->boolStates), tempe(p->temp), pStrain(p->pStrain), poissonsStrainInvalid(p->poissonsStrainInvalid),
      previousDt(p->previousDt), phaseOffset(p->phaseOffset), isDetached(p->isDetached), baseCiliaForce(p->baseCiliaForce),
      shiftCiliaForce(p->shiftCiliaForce) {
    _voxel = p;

    for (int i = 0; i < k->num_d_voxelMats; i++) {
        if (k->h_voxelMats[i] == p->mat) {
            mat = &k->d_voxelMats[i];
            break;
        }
    }

    for (unsigned i = 0; i < 6; i++) {
        if (p->links[i]) {
            links[i] = k->h_lookup_links[p->links[i]];
            // for (int j=0;j<k->num_d_links;j++) {
            // 	if (p->links[i] == k->h_links[j]) {
            // 		links[i] = &k->d_links[j];
            // 		continue;
            // 	}
            // }
        } else {
            links[i] = NULL;
        }
    }

    // mat = new VX3_MaterialVoxel(p->mat);
    if (p->ext) {
        VcudaMalloc((void **)&ext, sizeof(VX3_External));
        VX3_External temp2(p->ext);
        VcudaMemcpy(ext, &temp2, sizeof(VX3_External), VcudaMemcpyHostToDevice);
    } else {
        ext = NULL;
    }
}

VX3_Voxel::~VX3_Voxel() {
    if (ext) {
        MycudaFree(ext);
        ext = NULL;
    }
}

__device__ void VX3_Voxel::deviceInit(VX3_VoxelyzeKernel* k) {
    d_kernel = k;
    d_signal.value = 0;
    d_signal.activeTime = 0;

    d_signals.clear();

    // orient = VX3_Quat3D<>(); // default orientation
    targetPos = VX3_Vec3D<>();
    settleForceZ = 0;
    enableAttach = true;
    nonStickTimer = 0.0;

    // init randomState
    int randIndex = ix + k->WorldSize*iy + k->WorldSize*k->WorldSize*iz;
    // curand_init(k->RandomSeed + k->currentTime, randIndex, 0, &randomState);
    curand_init(k->RandomSeed, randIndex, 0, &randomState);

    WallRadius = (k->WorldSize - 1) * k->voxSize;
    WallForce = k->WallForce;

    // init linkdir
    for (int i=0;i<6;i++) {
        if (links[i]) {
            
            unbreakable = true; // sam: don't break voxels that are initialized with links

            if (links[i]->pVNeg==this) {
                links[i]->linkdirNeg = (linkDirection)i;
            } else if (links[i]->pVPos==this) {
                links[i]->linkdirPos = (linkDirection)i;
            } else {
                printf("linkdir initialization error.\n");
            }
        }
    }
    // d_group = new VX3_VoxelGroup(d_kernel); Sida: use halloc for now. note halloc and new need different de-allocation.
    d_group = (VX3_VoxelGroup*) hamalloc(sizeof(VX3_VoxelGroup));
    PRINT(d_kernel, "Create VoxelGroup (%p) in deviceInit.\n", d_group);
    if (d_group==NULL) {
        printf("halloc: Out of memory. Please increate the size of memory that halloc manages.\n");
    }
    d_group->deviceInit(d_kernel);
    //
    d_kernel->d_voxel_to_update_group.push_back(this);
    d_group->d_voxels.push_back(this);
    d_kernel->d_voxelgroups.push_back(d_group);

    enableFloor(d_kernel->enableFloor);
}
__device__ VX3_Voxel *VX3_Voxel::adjacentVoxel(linkDirection direction) const {
    VX3_Link *pL = links[(int)direction];
    if (pL)
        return pL->voxel(true) == this ? pL->voxel(false) : pL->voxel(true);
    else
        return NULL;
}

__device__ void VX3_Voxel::addLinkInfo(linkDirection direction, VX3_Link *link) {
    links[direction] = link;
    updateSurface();
}

__device__ void VX3_Voxel::removeLinkInfo(linkDirection direction) {
    links[direction] = NULL;
    updateSurface();
}

__device__ void VX3_Voxel::replaceMaterial(VX3_MaterialVoxel *newMaterial) {
    if (newMaterial != NULL) {

        linMom *= newMaterial->_mass / mat->_mass; // adjust momentums to keep velocity constant across material change
        angMom *= newMaterial->_momentInertia / mat->_momentInertia;
        setFloorStaticFriction(false);
        poissonsStrainInvalid = true;

        mat = newMaterial;
    }
}

__device__ bool VX3_Voxel::isYielded() const {
    for (int i = 0; i < 6; i++) {
        if (links[i] && links[i]->isYielded())
            return true;
    }
    return false;
}

__device__ bool VX3_Voxel::isFailed() const {
    for (int i = 0; i < 6; i++) {
        if (links[i] && links[i]->isFailed())
            return true;
    }
    return false;
}

__device__ void VX3_Voxel::setTemperature(float temperature) {
    tempe = temperature;
    for (int i = 0; i < 6; i++) {
        if (links[i] != NULL)
            links[i]->updateRestLength();
    }
}

__device__ VX3_Vec3D<float> VX3_Voxel::externalForce() {
    VX3_Vec3D<float> returnForce(ext->force());
    if (ext->isFixed(X_TRANSLATE) || ext->isFixed(Y_TRANSLATE) || ext->isFixed(Z_TRANSLATE)) {
        VX3_Vec3D<float> thisForce = (VX3_Vec3D<float>)-force();
        if (ext->isFixed(X_TRANSLATE))
            returnForce.x = thisForce.x;
        if (ext->isFixed(Y_TRANSLATE))
            returnForce.y = thisForce.y;
        if (ext->isFixed(Z_TRANSLATE))
            returnForce.z = thisForce.z;
    }
    return returnForce;
}

__device__ VX3_Vec3D<float> VX3_Voxel::externalMoment() {
    VX3_Vec3D<float> returnMoment(ext->moment());
    if (ext->isFixed(X_ROTATE) || ext->isFixed(Y_ROTATE) || ext->isFixed(Z_ROTATE)) {
        VX3_Vec3D<float> thisMoment = (VX3_Vec3D<float>)-moment();
        if (ext->isFixed(X_ROTATE))
            returnMoment.x = thisMoment.x;
        if (ext->isFixed(Y_ROTATE))
            returnMoment.y = thisMoment.y;
        if (ext->isFixed(Z_ROTATE))
            returnMoment.z = thisMoment.z;
    }
    return returnMoment;
}

__device__ VX3_Vec3D<float> VX3_Voxel::cornerPosition(voxelCorner corner) const {
    return (VX3_Vec3D<float>)pos + orient.RotateVec3D(cornerOffset(corner));
}

__device__ VX3_Vec3D<float> VX3_Voxel::cornerOffset(voxelCorner corner) const {
    VX3_Vec3D<> strains;
    for (int i = 0; i < 3; i++) {
        bool posLink = corner & (1 << (2 - i)) ? true : false;
        VX3_Link *pL = links[2 * i + (posLink ? 0 : 1)];
        if (pL && !pL->isFailed()) {
            strains[i] = (1 + pL->axialStrain(posLink)) * (posLink ? 1 : -1);
        } else
            strains[i] = posLink ? 1.0 : -1.0;
    }

    return (0.5 * baseSize()).Scale(strains);
}

// http://klas-physics.googlecode.com/svn/trunk/src/general/Integrator.cpp (reference)
__device__ void VX3_Voxel::timeStep(double dt, double currentTime, VX3_VoxelyzeKernel *k) {
    if (freezeAllVoxelsAfterAttach) {
        if (k->d_attach_manager->totalLinksFormed>=1) {
            return;
            //freeze all voxels when new link forms. just for a snapshot to analyze the position and angles.
        }
    }
    previousDt = dt;
    if (dt == 0.0f)
        return;

    if (ext && ext->isFixedAll()) {

        pos = originalPosition() + ext->translation();
        orient = ext->rotationQuat();
        haltMotion();
        return;
    }

    // Translation
    VX3_Vec3D<double> curForce = force();

    // Apply Force Field
    curForce.x += k->force_field.x_forcefield(pos.x, pos.y, pos.z, k->collisionCount, currentTime, k->recentAngle, k->targetCloseness,
                                              k->numClosePairs, k->num_d_voxels);
    curForce.y += k->force_field.y_forcefield(pos.x, pos.y, pos.z, k->collisionCount, currentTime, k->recentAngle, k->targetCloseness,
                                              k->numClosePairs, k->num_d_voxels);
    curForce.z += k->force_field.z_forcefield(pos.x, pos.y, pos.z, k->collisionCount, currentTime, k->recentAngle, k->targetCloseness,
                                              k->numClosePairs, k->num_d_voxels);

    VX3_Vec3D<double> fricForce = curForce;

    if (isFloorEnabled()) {
        floorForce(dt, &curForce); // floor force needs dt to calculate threshold to "stop" a slow voxel into static friction.
    }
    fricForce = curForce - fricForce;

    assert(!(curForce.x != curForce.x) || !(curForce.y != curForce.y) || !(curForce.z != curForce.z)); // assert non QNAN
    linMom += curForce * dt;
    //damp the giggling after new link formed.
    if (d_group->hasNewLink) {
        linMom = linMom * 0.99; // TODO: keep damping until damped???
    }

    VX3_Vec3D<double> translate(linMom * (dt * mat->_massInverse)); // movement of the voxel this timestep

    //	we need to check for friction conditions here (after calculating the translation) and stop things accordingly
    if (isFloorEnabled() &&
        floorPenetration() >=
            0) { // we must catch a slowing voxel here since it all boils down to needing access to the dt of this timestep.

        double work = fricForce.x * translate.x + fricForce.y * translate.y;                // F dot disp
        double hKe = 0.5 * mat->_massInverse * (linMom.x * linMom.x + linMom.y * linMom.y); // horizontal kinetic energy
        if (hKe + work <= 0)
            setFloorStaticFriction(true); // this checks for a change of direction according to the work-energy principle

        if (isFloorStaticFriction()) { // if we're in a state of static friction, zero out all horizontal motion
            linMom.x = linMom.y = 0;
            translate.x = translate.y = 0;
        }
    } else
        setFloorStaticFriction(false);

    pos += translate;

    // Rotation
    VX3_Vec3D<> curMoment = moment();
    angMom += curMoment * dt;

    // // damp the giggling after new link formed.
    if (d_group->hasNewLink) {
        angMom = angMom * 0.99; // TODO: keep damping until damped???
    }

    orient = VX3_Quat3D<>(angMom * (dt * mat->_momentInertiaInverse)) * orient; // update the orientation
    if (ext) {
        double size = mat->nominalSize();
        if (ext->isFixed(X_TRANSLATE)) {
            pos.x = ix * size + ext->translation().x;
            linMom.x = 0;
        }
        if (ext->isFixed(Y_TRANSLATE)) {
            pos.y = iy * size + ext->translation().y;
            linMom.y = 0;
        }
        if (ext->isFixed(Z_TRANSLATE)) {
            pos.z = iz * size + ext->translation().z;
            linMom.z = 0;
        }
        if (ext->isFixedAnyRotation()) { // if any rotation fixed, all are fixed
            if (ext->isFixedAllRotation()) {
                orient = ext->rotationQuat();
                angMom = VX3_Vec3D<double>();
            } else { // partial fixes: slow!
                VX3_Vec3D<double> tmpRotVec = orient.ToRotationVector();
                if (ext->isFixed(X_ROTATE)) {
                    tmpRotVec.x = 0;
                    angMom.x = 0;
                }
                if (ext->isFixed(Y_ROTATE)) {
                    tmpRotVec.y = 0;
                    angMom.y = 0;
                }
                if (ext->isFixed(Z_ROTATE)) {
                    tmpRotVec.z = 0;
                    angMom.z = 0;
                }
                orient.FromRotationVector(tmpRotVec);
            }
        }
    }
    //	we need to check for friction conditions here (after calculating the translation) and stop things accordingly
    if (isFloorEnabled() && floorPenetration() >= 0) {
        // we must catch a slowing voxel here since it all boils down to needing access to the dt of this timestep.
        if (isFloorStaticFriction()) {
            angMom = VX3_Vec3D<>(0, 0, 0);
        }
    }

    poissonsStrainInvalid = true;

    // if (d_signals.size()==1) printf("< %p, %f, %s, %d, %d\n", this, currentTime, k->vxa_filename, d_signals.sizeof_chunk,
    // d_signals.size());

    if (k->EnableSignals) {
        // printf("%f) before propagateSignal. this=%p.\n",currentTime, this);
        propagateSignal(currentTime);
        packMaker(currentTime);
        if (mat->signalValueDecay <= 1.0) // sam: hack to keep material lit (set decay > 1)
            localSignalDecay(currentTime);
    }


    // // sam:
    // if (teleportPos.Length2() > 0) {
    //     pos = teleportPos;
    //     // orient = teleportOrient;
    //     // angMom = VX3_Vec3D<>(0, 0, 0);
    //     // linMom = VX3_Vec3D<>(0, 0, 0);
    // }


}

__device__ void VX3_Voxel::localSignalDecay(double currentTime) {
    if (localSignaldt > currentTime)
        return;
    if (localSignal < 0.1) {
        // lower than threshold, simply ignore.
        localSignal = 0;
    } else {
        localSignal = localSignal * 0.9;
        localSignaldt = currentTime + 0.01;
    }
}

__device__ void VX3_Voxel::packMaker(double currentTime) {
    if (!mat->isPaceMaker)
        return;
    if (packmakerNextPulse > currentTime)
        return;

    receiveSignal(100, currentTime, true);
    packmakerNextPulse = currentTime + mat->PaceMakerPeriod;
}

__device__ void VX3_Voxel::receiveSignal(double signalValue, double activeTime, bool force) {
    if (!force) {
        if (inactiveUntil > activeTime)
            return;
    }
    if (signalValue < 0.1) {
        // lower than threshold, simply ignore.
        return;
    }

    // if received a signal, this cell will activate at activeTime, and before that, no need to receive another signal.
    inactiveUntil = activeTime + mat->inactivePeriod;

    localSignal = signalValue;
    // VX3_Signal *s = new VX3_Signal();
    d_signal.value = signalValue * mat->signalValueDecay;
    if (d_signal.value < 0.1)
        d_signal.value = 0;
    d_signal.activeTime = activeTime;
    // d_signals.push_back(s);
    // InsertSignalQueue(signalValue, currentTime + mat->signalTimeDelay);
}
__device__ void VX3_Voxel::propagateSignal(double currentTime) {
    // first one in queue, check the time
    // if (inactiveUntil > currentTime)
    //     return;
    if (d_signal.activeTime > currentTime)
        return;
    if (d_signal.value < 0.1) {
        return;
    }
    for (int i = 0; i < 6; i++) {
        if (links[i]) {
            if (links[i]->pVNeg == this) {
                links[i]->pVPos->receiveSignal(d_signal.value, currentTime + mat->signalTimeDelay, false);
            } else {
                links[i]->pVNeg->receiveSignal(d_signal.value, currentTime + mat->signalTimeDelay, false);
            }
        }
    }

    d_signal.value=0;
    d_signal.activeTime=0;
    inactiveUntil = currentTime + 2 * mat->signalTimeDelay + mat->inactivePeriod;
    // if (s)
    //     delete s;
    //     // printf("%f) delete s. this=%p. d_signals.size() %d. \n",currentTime, this, d_signals.size() );
}

__device__ VX3_Vec3D<double> VX3_Voxel::force() {

    // forces from internal bonds
    VX3_Vec3D<double> totalForce(0, 0, 0);
    for (int i = 0; i < 6; i++) {
        if (links[i]) {
            totalForce += links[i]->force(this); // total force in LCS
        }
    }
    totalForce = orient.RotateVec3D(totalForce); // from local to global coordinates
    assert(!(totalForce.x != totalForce.x) || !(totalForce.y != totalForce.y) || !(totalForce.z != totalForce.z)); // assert non QNAN

    // other forces
    if (externalExists())
        totalForce += external()->force();                     // external forces
        
    // totalForce -= velocity() * mat->globalDampingTranslateC(); // global damping f-cv
    totalForce -= velocity() * mat->globalDampingTranslateC() * mat->SlowDampingFrac; // sam

    totalForce.z += mat->gravityForce();                       // gravity, according to f=mg

    // no collision yet
    // if (isCollisionsEnabled()){
    // for (int i=0;i<colWatch.size();i++){
    // }
    // }
    totalForce -= contactForce;
    contactForce.clear();

    // sam:
    if (mat->LockZ) {
        CiliaForce.z = 0;
        if ( (!d_kernel->firstRound)  && (!unbreakable))  // not removeMat
            CiliaForce *= d_kernel->CiliaFracAfterFirstRound; 
    }

    CiliaForce *= d_group->d_voxels.size(); // sam

    totalForce += CiliaForce * mat->Cilia;
    CiliaForce.clear();

    // sam:
    if ( (targetPos.Length2() > 0) && (!weakLink) ) {  // was detached
        totalForce += (targetPos - pos);
    }

    if (settleForceZ < 0)
        totalForce.z += settleForceZ;

    // sam
    if ( (mat->WaterLevel > 0) and (pos.z >= mat->WaterLevel*d_kernel->voxSize) ) { 
        double adjustment = pos.z / d_kernel->voxSize - mat->WaterLevel;
        totalForce.z += adjustment*adjustment * mat->gravityForce(); 
    }

    // sam: let's test a square first
    if (WallForce > 0 && WallRadius > 0) {
        double adjustX = 0;
        double adjustY = 0;
        if (pos.x >= WallRadius) {
            adjustX = -1 * (WallRadius - pos.x) * (WallRadius - pos.x);
        } else if (pos.x <= 0) {
            adjustX = pos.x*pos.x;
        }
        if (pos.y >= WallRadius) {
            adjustY = -1 * (WallRadius - pos.y) * (WallRadius - pos.y);
        } else if (pos.y <= 0) {
            adjustY = pos.y*pos.y;
        }
        totalForce.x += adjustX * WallForce;
        totalForce.y += adjustY * WallForce;
    }

    // sam
    // totalForce.z += mat->Buoyancy * mat->mass();  // sam: lift bodies without simulating light stiff material
    if (mat->Buoyancy > 0) {
        totalForce.z += -1 * mat->gravityForce();  // just remove gravity
    }

    // sam
    if (mat->LockZ) {
        totalForce.z = 0;
    }

    return totalForce;
}

__device__ VX3_Vec3D<double> VX3_Voxel::moment() {
    // moments from internal bonds
    VX3_Vec3D<double> totalMoment(0, 0, 0);
    for (int i = 0; i < 6; i++) {
        if (links[i]) {
            totalMoment += links[i]->moment(this); // total force in LCS
        }
    }
    totalMoment = orient.RotateVec3D(totalMoment);

    // other moments
    if (externalExists())
        totalMoment += external()->moment();                        // external moments
    // totalMoment -= angularVelocity() * mat->globalDampingRotateC(); // global damping
    totalMoment -= angularVelocity() * mat->globalDampingRotateC() * mat->SlowDampingFrac; // sam

    // sam
    if (mat->LockZ) { 
        totalMoment.x = 0;
        totalMoment.y = 0;
    }

    // sam
    if (mat->NoSpin) {
        totalMoment.x = 0;
        totalMoment.y = 0; 
        totalMoment.z = 0;
    }

    return totalMoment;
}

__device__ void VX3_Voxel::floorForce(float dt, VX3_Vec3D<double> *pTotalForce) {
    float CurPenetration = floorPenetration(); // for now use the average.
    if (CurPenetration >= 0) {
        VX3_Vec3D<double> vel = velocity();
        VX3_Vec3D<double> horizontalVel(vel.x, vel.y, 0);

        float normalForce = mat->penetrationStiffness() * CurPenetration;

        pTotalForce->z += normalForce - mat->collisionDampingTranslateC() * vel.z; // in the z direction: k*x-C*v - spring and damping

        if (isFloorStaticFriction()) { // If this voxel is currently in static friction mode (no lateral motion)
            assert(horizontalVel.Length2() == 0);
            float surfaceForceSq =
                (float)(pTotalForce->x * pTotalForce->x + pTotalForce->y * pTotalForce->y); // use squares to avoid a square root
            float frictionForceSq = (mat->muStatic * normalForce) * (mat->muStatic * normalForce);

            if (surfaceForceSq > frictionForceSq)
                setFloorStaticFriction(false); // if we're breaking static friction, leave the forces as they currently have been calculated
                                               // to initiate motion this time step
        } else { // even if we just transitioned don't process here or else with a complete lack of momentum it'll just go back to static
                 // friction
            *pTotalForce -=
                mat->muKinetic * normalForce * horizontalVel.Normalized(); // add a friction force opposing velocity according to the normal
                                                                           // force and the kinetic coefficient of friction
        }
    } else
        setFloorStaticFriction(false);
}

__device__ VX3_Vec3D<float> VX3_Voxel::strain(bool poissonsStrain) const {
    // if no connections in the positive and negative directions of a particular axis, strain is zero
    // if one connection in positive or negative direction of a particular axis, strain is that strain - ?? and force or constraint?
    // if connections in both the positive and negative directions of a particular axis, strain is the average.

    VX3_Vec3D<float> intStrRet(0, 0, 0); // intermediate strain return value. axes according to linkAxis enum
    int numBondAxis[3] = {0};            // number of bonds in this axis (0,1,2). axes according to linkAxis enum
    bool tension[3] = {false};
    for (int i = 0; i < 6; i++) { // cycle through link directions
        if (links[i]) {
            int axis = toAxis((linkDirection)i);
            intStrRet[axis] += links[i]->axialStrain(isNegative((linkDirection)i));
            numBondAxis[axis]++;
        }
    }
    for (int i = 0; i < 3; i++) { // cycle through axes
        if (numBondAxis[i] == 2)
            intStrRet[i] *= 0.5f; // average
        if (poissonsStrain) {
            tension[i] = ((numBondAxis[i] == 2) ||
                          (ext && (numBondAxis[i] == 1 &&
                                   (ext->isFixed((dofComponent)(1 << i)) ||
                                    ext->force()[i] != 0)))); // if both sides pulling, or just one side and a fixed or forced voxel...
        }
    }

    if (poissonsStrain) {
        if (!(tension[0] && tension[1] && tension[2])) { // if at least one isn't in tension
            float add = 0;
            for (int i = 0; i < 3; i++)
                if (tension[i])
                    add += intStrRet[i];
            float value = pow(1.0f + add, -mat->poissonsRatio()) - 1.0f;
            for (int i = 0; i < 3; i++)
                if (!tension[i])
                    intStrRet[i] = value;
        }
    }

    return intStrRet;
}

__device__ VX3_Vec3D<float> VX3_Voxel::poissonsStrain() {
    if (poissonsStrainInvalid) {
        pStrain = strain(true);
        poissonsStrainInvalid = false;
    }
    return pStrain;
}

__device__ float VX3_Voxel::transverseStrainSum(linkAxis axis) {
    if (mat->poissonsRatio() == 0)
        return 0;

    VX3_Vec3D<float> psVec = poissonsStrain();

    switch (axis) {
    case X_AXIS:
        return psVec.y + psVec.z;
    case Y_AXIS:
        return psVec.x + psVec.z;
    case Z_AXIS:
        return psVec.x + psVec.y;
    default:
        return 0.0f;
    }
}

__device__ float VX3_Voxel::transverseArea(linkAxis axis) {
    float size = (float)mat->nominalSize();
    if (mat->poissonsRatio() == 0)
        return size * size;

    VX3_Vec3D<> psVec = poissonsStrain();

    switch (axis) {
    case X_AXIS:
        return (float)(size * size * (1 + psVec.y) * (1 + psVec.z));
    case Y_AXIS:
        return (float)(size * size * (1 + psVec.x) * (1 + psVec.z));
    case Z_AXIS:
        return (float)(size * size * (1 + psVec.x) * (1 + psVec.y));
    default:
        return size * size;
    }
}

__device__ void VX3_Voxel::updateSurface() {
    bool interior = true;
    for (int i = 0; i < 6; i++)
        if (!links[i]) {
            interior = false;
        } else if (links[i]->isDetached) {
            interior = false;
        }
    interior ? boolStates |= SURFACE : boolStates &= ~SURFACE;
}

__device__ void VX3_Voxel::enableCollisions(bool enabled, float watchRadius) {
    enabled ? boolStates |= COLLISIONS_ENABLED : boolStates &= ~COLLISIONS_ENABLED;
}

__device__ void VX3_Voxel::generateNearby(int linkDepth, int gindex, bool surfaceOnly) {
    assert(false); // not used. near by has logic flaws.
}

// __device__ void VX3_Voxel::updateGroup() {

// }

// __device__ void VX3_Voxel::switchGroupTo(VX3_VoxelGroup* group) {
//     if (d_group==group)
//         return;
//     if (d_group) {
//         // TODO: check all memory in that group is freed if necessary.
//         // use delete because this is created by new. (new and delete, malloc and free)
//         // VX3_VoxelGroup* to_delete = d_group; // because d_group->switchAllVoxelsTo() will change the pointer d_group, so save it here for deletion.
//         d_group->switchAllVoxelsTo(group);
//         // delete to_delete;
//         // Free this memory seems will spend a lot of time checking conditions, just leave it there for now.
//         // d_group = group;
//     } else {
//         d_group = group;
//     }
// }

__device__ void VX3_Voxel::changeOrientationTo(VX3_Quat3D<> q) {
    baseCiliaForce = q.RotateVec3DInv(orient.RotateVec3D(baseCiliaForce));
    shiftCiliaForce = q.RotateVec3DInv(orient.RotateVec3D(shiftCiliaForce));
    orient = q;
}

__device__ void VX3_Voxel::isSingletonOrSmallBar(bool *isSingleton, bool *isSmallBar, int *SmallBarDirection) {
    int direction = -1;
    for (int i=0;i<6;i++) {
        if (links[i]) {
            direction = i;
            break;
        }
    }
    if (direction==-1) {
        *isSingleton = true;
        *isSmallBar = false;
    } else if (d_group->d_voxels.size() == 2) {
        *isSingleton = false;
        *isSmallBar = true;
        *SmallBarDirection = direction;
    } else {
        *isSingleton = false;
        *isSmallBar = false;
    }
}

__device__ void VX3_Voxel::grow() {
    double p = (double) d_kernel->randomGenerator->randint(1000) * 0.001; //TODO: Oh, this generate the same "random" number for all threads!!
    // Solution: change randomGenerator to preallocate a bunch of random numbers, and get them by atomic operation.
    printf("p %f. d_kernel->SurfaceGrowth_Rate %f. \n", p, d_kernel->SurfaceGrowth_Rate);
    if (p > d_kernel->SurfaceGrowth_Rate)
        return;
    
    int num_directions = 0;
    int available_directions[6];
    VX3_Vec3D<> available_position[6];

    for (int i=0;i<6;i++) {
        if (links[i]==NULL) { // available link slot
            available_position[num_directions] = potentialNeighborPosition(i);
            if (available_position[num_directions].z > d_kernel->voxSize / 2 && d_kernel->enableFloor) { // and above ground
                available_directions[num_directions] = i;
                num_directions ++;
            }
        }
    }
    if (num_directions==0)
        return;
    int choice = d_kernel->randomGenerator->randint(num_directions);
    printf("grow()\n");
    d_kernel->d_growth_manager->grow(this, available_directions[choice], available_position[choice]);
}

__device__ VX3_Vec3D<> VX3_Voxel::potentialNeighborPosition(int linkdir) {
    VX3_Vec3D<> pnPos = VX3_Vec3D<>();
    switch ((linkDirection)linkdir) {
    case X_POS:
        pnPos.x += d_kernel->voxSize;
        break;
    case X_NEG:
        pnPos.x -= d_kernel->voxSize;
        break;
    case Y_POS:
        pnPos.y += d_kernel->voxSize;
        break;
    case Y_NEG:
        pnPos.y -= d_kernel->voxSize;
        break;
    case Z_POS:
        pnPos.z += d_kernel->voxSize;
        break;
    case Z_NEG:
        pnPos.z -= d_kernel->voxSize;
    }
    // real coordinate
    pnPos = pos + orient.RotateVec3D(pnPos);
    return pnPos;
}