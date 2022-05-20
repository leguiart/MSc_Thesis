//
// Created by David Matthews on 5/23/20.
//
//

#ifndef COLLISIONDETECTIONSYSTEM_TESTSYSTEMHEURISTIC_CUH
#define COLLISIONDETECTIONSYSTEM_TESTSYSTEMHEURISTIC_CUH

#include "../include/CollisionSystem.cuh"

const unsigned int levels[] = {2, 3, 7, 11, 21, 41, 81, 128, 255, 511, 1024, 2048, 5000, 10000, 11000};
const int MAX_COLS_PER_OBJECT = 0;
const unsigned int MAX_LOOPS = 1024;

bool testSmallSystems(int verbose);

__global__ void initKernel(CollisionSystem *colSys);

__global__ void find_cols_tree_Kernel(CollisionSystem *colSys);
__global__ void find_cols_N2_Kernel(CollisionSystem *colSys);
__global__ void assert_collision(CollisionSystem *colSys, bool expectCollision, float pX, float pY, float pZ, float pR);

bool testSmallSystemsFromGPU(int verbose);

bool testN2CollisionSystemHeuristic(int verbose);

/**
 * Run tests on the CollisionSystem in a Heuristic manner.
 *
 * @param verbose Verbosity level. 0 = no printing, 1 = progress and errors, 2 or greater = print all debugging info.
 *
 * @return True if passed all tests, false otherwise.
 */
bool testTreeCollisionSystemHeuristic(int verbose);


#endif //COLLISIONDETECTIONSYSTEM_TESTSYSTEMHEURISTIC_CUH
