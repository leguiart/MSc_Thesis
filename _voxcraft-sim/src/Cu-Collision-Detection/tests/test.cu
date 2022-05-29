#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>

#include "../include/Morton.cuh"
#include "testSystemHeuristic.cuh"

bool runMortonNumberTestOnIntCPU();
bool runMortonNumberTestOnIntGPU();
bool runMortonNumberTestOnIntGPUFromGPU();

__global__
void mortonNumberKernel(int num_items, unsigned long long int *testInts, unsigned long long int *resultInts);

bool runMortonNumberTestOnFloat();
bool runSmallSystemTests();
bool runSmallSystemTestsFromGPU();
bool runN2SystemHeuristicTest();
bool runTreeSystemHeuristicTest();

int main(int argc, char *argv[]) {

    bool passedAllTests = true;

    if (passedAllTests)
        passedAllTests &= runMortonNumberTestOnIntCPU();

    if (passedAllTests)
        passedAllTests &= runMortonNumberTestOnIntGPU();

    if (passedAllTests)
        passedAllTests &= runMortonNumberTestOnIntGPUFromGPU();

    if (passedAllTests)
        passedAllTests &= runMortonNumberTestOnFloat();

    if (passedAllTests)
        passedAllTests &= runSmallSystemTests();

    if (passedAllTests)
        passedAllTests &= runSmallSystemTestsFromGPU();

    if (passedAllTests)
        passedAllTests &= runN2SystemHeuristicTest();

    if (passedAllTests) {
        if (argc > 1) {
            std::cout << "Running Tree Heuristic Tests..." << std::endl;
            passedAllTests &= runTreeSystemHeuristicTest();
        } else {
            std::cout << "Skipping Tree Heuristic Tests" << std::endl;
        }
    }

    if (passedAllTests) {
        std::cout << "Passed all tests." << std::endl;
    }

    return 0;
}

struct expandBits64_functor {
    __device__
    unsigned long long int operator()(unsigned long long int input) {
        return expandBits64(input);
    }
};

bool runMortonNumberTestOnIntCPU() {
    std::cout << "Testing Morton Number Generation on Ints on CPU ... " << std::flush;

    unsigned long long int testInts[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xF, 0x1FFFFF, // <= 21 bits used
                                         0x2FFFFF, 0xFFFFF1, 0xFFFFF2}; // > 21 bits used, take most sig bits.

    unsigned long long int testIntsValidResults[] = {0x0, 0x1, 0x8, 0x9, 0x40, 0x41, 0x249, 0x1249249249249249,
                                                     0x1049249249249249, 0x1249249249249248, 0x1249249249249248,};

    for (int testIdx = 0; testIdx < sizeof(testInts) / sizeof(unsigned long long int); testIdx++) {
        unsigned long long int currentResult = expandBits64(testInts[testIdx]);
        if (currentResult != testIntsValidResults[testIdx]) {
            printf(" Failed on %llu which should have been %llu but was %llu\n",
                    testInts[testIdx],
                    testIntsValidResults[testIdx],
                    currentResult);
            return false;
        }
    }

    printf(" Passed %lu tests\n", sizeof(testInts) / sizeof(unsigned long long int));
    return true;
}

bool runMortonNumberTestOnIntGPU() {
    std::cout << "Testing Morton Number Generation on Ints on GPU ... " << std::flush;

    unsigned long long int testInts[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xF, 0x1FFFFF, // <= 21 bits used
                                         0x2FFFFF, 0xFFFFF1, 0xFFFFF2}; // > 21 bits used, take most sig bits.

    unsigned long long int testIntsValidResults[] = {0x0, 0x1, 0x8, 0x9, 0x40, 0x41, 0x249, 0x1249249249249249,
                                                     0x1049249249249249, 0x1249249249249248, 0x1249249249249248,};

    int NUM_TESTS = sizeof(testInts) / sizeof(unsigned long long int);

    thrust::device_vector<unsigned long long int> testInts_d(NUM_TESTS);

    thrust::copy(testInts, testInts + NUM_TESTS, testInts_d.begin());
    thrust::device_vector<unsigned long long int> testResults_d(NUM_TESTS);

    thrust::transform(testInts_d.begin(),
            testInts_d.end(),
            testResults_d.begin(),
            expandBits64_functor());

    for (int testIdx = 0; testIdx < sizeof(testInts) / sizeof(unsigned long long int); testIdx++) {
        unsigned long long int currentResult = testResults_d[testIdx];
        if (currentResult != testIntsValidResults[testIdx]) {
            printf(" Failed on %llu which should have been %llu but was %llu\n",
                    testInts[testIdx],
                    testIntsValidResults[testIdx],
                    currentResult);
            return false;
        }
    }

    printf(" Passed %lu tests\n", sizeof(testInts) / sizeof(unsigned long long int));
    return true;
}

__global__
void mortonNumberKernel(int num_items, unsigned long long int *testInts, unsigned long long int *resultInts) {
    if (threadIdx.x + blockDim.x * threadIdx.x < 1) {
        thrust::transform(thrust::device, testInts,
                testInts + num_items,
                resultInts,
                expandBits64_functor());
    }
}

bool runMortonNumberTestOnIntGPUFromGPU() {
    std::cout << "Testing Morton Number Generation on Ints on GPU from GPU ... " << std::flush;

    unsigned long long int testInts[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xF, 0x1FFFFF, // <= 21 bits used
                                         0x2FFFFF, 0xFFFFF1, 0xFFFFF2}; // > 21 bits used, take most sig bits.

    unsigned long long int testIntsValidResults[] = {0x0, 0x1, 0x8, 0x9, 0x40, 0x41, 0x249, 0x1249249249249249,
                                                     0x1049249249249249, 0x1249249249249248, 0x1249249249249248,};

    int NUM_TESTS = sizeof(testInts) / sizeof(unsigned long long int);

    thrust::device_vector<unsigned long long int> testInts_d(NUM_TESTS);

    thrust::copy(testInts, testInts + NUM_TESTS, testInts_d.begin());
    thrust::device_vector<unsigned long long int> testResults_d(NUM_TESTS);

    mortonNumberKernel<<<1,1>>>(NUM_TESTS, thrust::raw_pointer_cast(testInts_d.data()), thrust::raw_pointer_cast(testResults_d.data()));

    for (int testIdx = 0; testIdx < sizeof(testInts) / sizeof(unsigned long long int); testIdx++) {
        unsigned long long int currentResult = testResults_d[testIdx];
        if (currentResult != testIntsValidResults[testIdx]) {
            printf(" Failed on %llu which should have been %llu but was %llu\n",
                    testInts[testIdx],
                    testIntsValidResults[testIdx],
                    currentResult);
            return false;
        }
    }

    printf(" Passed %lu tests\n", sizeof(testInts) / sizeof(unsigned long long int));
    return true;
}

bool runMortonNumberTestOnFloat() {
    std::cout << "Testing Karras Morton Number Generation on Floats ... " << std::flush;

    float3 testFloats[] = {{0, 0, 0},
                           {0, 0, 1},
                           {0, 1, 0},
                           {1, 0, 0},
                           {1, 1, 0},
                           {1, 1, 1}};
    unsigned long long int testFloatsValidResults[] = {0, 0x4924924924924924, 0x2492492492492492, 0x1249249249249249,
                                                       0x36db6db6db6db6db, 0x7fffffffffffffff};

    float3 pt;
    unsigned long long mPts;
    unsigned int mIds;
    init_morton_func_fast mortonFloatsFunc = init_morton_func_fast({0, 1},
            {0, 1},
            {0, 1},
            &(pt.x),
            &(pt.y),
            &(pt.z),
            &mPts,
            &mIds);

    for (int testIdx = 0; testIdx < sizeof(testFloats) / sizeof(float3); testIdx++) {
        pt.x = testFloats[testIdx].x;
        pt.y = testFloats[testIdx].y;
        pt.z = testFloats[testIdx].z;

        mortonFloatsFunc(0);

        if (mPts != testFloatsValidResults[testIdx]) {
            std::cout << mPts << " " << testFloatsValidResults[testIdx] << " "
                      << mPts - testFloatsValidResults[testIdx];
            printf(" Failed on (%f, %f, %f) which should have been %llu but was %llu\n",
                    testFloats[testIdx].x,
                    testFloats[testIdx].y,
                    testFloats[testIdx].z,
                    testFloatsValidResults[testIdx],
                    mPts);
            return false;
        }
    }

    printf(" Passed %lu tests\n", sizeof(testFloats) / sizeof(float3));
    return true;
}

bool runSmallSystemTests() {
    bool didPass = testSmallSystems(1);
    std::cout << "\rTesting Small Collision Systems ... " << std::flush;

    if (didPass) {
        std::cout << " Passed 13 tests" << std::endl;
        return true;
    } else {
        std::cout << " Failed!" << std::endl;
        return false;
    }
}
bool runSmallSystemTestsFromGPU() {
    bool didPass = testSmallSystemsFromGPU(1);
    std::cout << "\rTesting Small Collision Systems from GPU ... " << std::flush;

    if (didPass) {
        std::cout << " Passed 13 tests" << std::endl;
        return true;
    } else {
        std::cout << " Failed!" << std::endl;
        return false;
    }
}

bool runN2SystemHeuristicTest() {
    bool didPass = testN2CollisionSystemHeuristic(1);

    std::cout << "\rTesting N2 Collision System Heuristics ... " << std::flush;

    if (didPass) {
        std::cout << " Passed" << std::endl;
        return true;
    } else {
        std::cout << " Failed!" << std::endl;
        return false;
    }
}

bool runTreeSystemHeuristicTest() {
    bool didPass = testTreeCollisionSystemHeuristic(2);

    std::cout << "\rTesting Tree Collision System Heuristics ... " << std::flush;

    if (didPass) {
        std::cout << " Passed" << std::endl;
        return true;
    } else {
        std::cout << " Failed!" << std::endl;
        return false;
    }
}