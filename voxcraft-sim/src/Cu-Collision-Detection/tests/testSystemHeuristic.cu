//
// Created by David Matthews on 5/23/20.
//
//
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "../include/CollisionSystem.cuh"
#include "../include/Collisions.cuh"

#include "testSystemHeuristic.cuh"


bool testSmallSystems(int verbose) {
    CollisionSystem *colSystem = new CollisionSystem(2, 1, true);
    auto x_pos_h = colSystem->get_x_pos_host();
    auto y_pos_h = colSystem->get_y_pos_host();
    auto z_pos_h = colSystem->get_z_pos_host();
    auto r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }
    colSystem->init();
    int numbCollisionsN2 = colSystem->find_collisions_N2();
    int numbCollisionsTree = colSystem->find_collisions();

    if (!colSystem->check_collisions(0.f, 0.f, 0.f, 1.f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (A.i). Expected a collision with test sphere but did not detect one." << std::endl;
            return false;
        }
    }
    if (colSystem->check_collisions(0.3f, 0.f, 0.f, .10f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (A.ii). Expected no collisions with test sphere but detected one." << std::endl;
            return false;
        }
    }
    if (!colSystem->check_collisions(0.3f, 0.f, 0.f, .10001f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (A.iii). Expected a collision with test sphere but did not detect one." << std::endl;
            return false;
        }
    }
    if (colSystem->check_collisions(10.f, 0.f, 0.f, 1.f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (A.iv). Expected a no collisions with test sphere but detected one." << std::endl;
            return false;
        }
    }
    delete colSystem;

    if (numbCollisionsN2 != 1) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 2 particle test (A). Expected 1 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }

    if (numbCollisionsTree != 1) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 2 particle test (A). Expected 1 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    colSystem = new CollisionSystem(2, 1, true);
    x_pos_h = colSystem->get_x_pos_host();
    y_pos_h = colSystem->get_y_pos_host();
    z_pos_h = colSystem->get_z_pos_host();
    r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = -0.1;
    x_pos_h[1] = 0.1;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.2;
    }
    colSystem->init();
    numbCollisionsN2 = colSystem->find_collisions_N2();
    numbCollisionsTree = colSystem->find_collisions();
    if (!colSystem->check_collisions(0.f, 0.f, 0.f, .0f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (B.i). Expected a collision with test sphere but did detect one." << std::endl;
            return false;
        }
    }
    if (colSystem->check_collisions(0.4f, 0.f, 0.f, .01f)) {
        if (verbose > 0) {
            std::cout << "check collision failed 2 particle test (B.ii). Expected no collisions with test sphere but detected one." << std::endl;
            return false;
        }
    }
    delete colSystem;


    if (numbCollisionsN2 != 1) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 2 particle test (B). Expected 1 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }

    if (numbCollisionsTree != 1) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 2 particle test (B). Expected 1 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    colSystem = new CollisionSystem(3, 1, true);
    x_pos_h = colSystem->get_x_pos_host();
    y_pos_h = colSystem->get_y_pos_host();
    z_pos_h = colSystem->get_z_pos_host();
    r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;
    x_pos_h[2] = 0.25;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }
    colSystem->init();
    numbCollisionsN2 = colSystem->find_collisions_N2();
    numbCollisionsTree = colSystem->find_collisions();
    delete colSystem;


    if (numbCollisionsN2 != 2) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 3 particle test (A). Expected 2 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }

    if (numbCollisionsTree != 2) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 3 particle test (A). Expected 2 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    colSystem = new CollisionSystem(3, 1, true);
    x_pos_h = colSystem->get_x_pos_host();
    y_pos_h = colSystem->get_y_pos_host();
    z_pos_h = colSystem->get_z_pos_host();
    r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;
    x_pos_h[2] = 0.11;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }
    colSystem->init();
    numbCollisionsN2 = colSystem->find_collisions_N2();
    numbCollisionsTree = colSystem->find_collisions();
    delete colSystem;

    if (numbCollisionsN2 != 3) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 3 particle test (B). Expected 2 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }

    if (numbCollisionsTree != 3) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 3 particle test (B). Expected 2 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    return true;
}

__global__ void initKernel(CollisionSystem *colSys) {
    if (threadIdx.x + blockDim.x * blockIdx.x < 1) {
        colSys->update_x_pos_ranks();
        colSys->update_y_pos_ranks();
        colSys->update_z_pos_ranks();
        colSys->update_mortons();
        colSys->build_tree();
        colSys->update_bounding_boxes();
    }
}

__global__ void find_cols_tree_Kernel(CollisionSystem *colSys) {
    if (threadIdx.x + blockDim.x * blockIdx.x < 1) {
        colSys->find_collisions_device();
    }
}

__global__ void find_cols_N2_Kernel(CollisionSystem *colSys) {
    if (threadIdx.x + blockDim.x * blockIdx.x < 1) {
        colSys->find_collisions_N2_device();
    }
}

__global__ void assert_collision(CollisionSystem *colSys, bool expectCollision, float pX, float pY, float pZ, float pR) {
    if (threadIdx.x + blockDim.x * blockIdx.x < 1) {
        assert(colSys->check_collisions_device(pX, pY, pZ, pR) == expectCollision);
    }
}

bool testSmallSystemsFromGPU(int verbose) {
    CollisionSystem *colSystem = new CollisionSystem(2, 1, true);
    CollisionSystem *colSystem_d;
    cudaMalloc((void**)&colSystem_d, sizeof(CollisionSystem));
    cudaMemcpy((void *) colSystem_d, (void * ) colSystem, sizeof(CollisionSystem), cudaMemcpyHostToDevice);

    auto x_pos_h = colSystem->get_x_pos_host();
    auto y_pos_h = colSystem->get_y_pos_host();
    auto z_pos_h = colSystem->get_z_pos_host();
    auto r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }

    colSystem->update_all_from_host();

    initKernel<<<1,1>>>(colSystem_d);
    find_cols_tree_Kernel<<<1,1>>>(colSystem_d);
    int numbCollisionsTree;
    cudaMemcpy((void*)&numbCollisionsTree, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();

    find_cols_N2_Kernel<<<1,1>>>(colSystem_d);
    int numbCollisionsN2;
    cudaMemcpy((void*)&numbCollisionsN2, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();

    assert_collision<<<1,1>>>(colSystem_d, true,  0.f, 0.f, 0.f, 1.f);
    assert_collision<<<1,1>>>(colSystem_d, false, 0.3f, 0.f, 0.f, .10f);
    assert_collision<<<1,1>>>(colSystem_d, true,  0.3f, 0.f, 0.f, .10001f);
    assert_collision<<<1,1>>>(colSystem_d, false, 10.f, 0.f, 0.f, 1.f);
    delete colSystem;

    cudaFree(colSystem_d);
    if (numbCollisionsN2 != 1) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 2 particle test. Expected 1 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }

    if (numbCollisionsTree != 1) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 2 particle test. Expected 1 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    // TEST 2
    colSystem = new CollisionSystem(3, 1, true);
    cudaMalloc((void**)&colSystem_d, sizeof(CollisionSystem));
    cudaMemcpy((void *) colSystem_d, (void * ) colSystem, sizeof(CollisionSystem), cudaMemcpyHostToDevice);

    x_pos_h = colSystem->get_x_pos_host();
    y_pos_h = colSystem->get_y_pos_host();
    z_pos_h = colSystem->get_z_pos_host();
    r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;
    x_pos_h[2] = 0.25;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }
    colSystem->update_all_from_host();
    initKernel<<<1,1>>>(colSystem_d);
    find_cols_tree_Kernel<<<1,1>>>(colSystem_d);

    cudaMemcpy((void*)&numbCollisionsTree, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();

    find_cols_N2_Kernel<<<1,1>>>(colSystem_d);
    cudaMemcpy((void*)&numbCollisionsN2, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();

    assert_collision<<<1,1>>>(colSystem_d, true,  0.f, 0.f, 0.f, 1.f);
    assert_collision<<<1,1>>>(colSystem_d, true,  0.f, 0.f, 0.f, 0.f);
    assert_collision<<<1,1>>>(colSystem_d, false,  0.4f, 0.f, 0.f, .01f);


    delete colSystem;
    cudaFree(colSystem_d);

    if (numbCollisionsN2 != 2) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 3 particle test (A). Expected 2 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }
    if (numbCollisionsTree != 2) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 3 particle test (A). Expected 2 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }

    colSystem = new CollisionSystem(3, 1, true);
    cudaMalloc((void**)&colSystem_d, sizeof(CollisionSystem));
    cudaMemcpy((void *) colSystem_d, (void * ) colSystem, sizeof(CollisionSystem), cudaMemcpyHostToDevice);

    x_pos_h = colSystem->get_x_pos_host();
    y_pos_h = colSystem->get_y_pos_host();
    z_pos_h = colSystem->get_z_pos_host();
    r_h = colSystem->get_radius_host();

    // init positions
    x_pos_h[0] = 0.0;
    x_pos_h[1] = 0.1;
    x_pos_h[2] = 0.11;

    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        y_pos_h[i] = 0.0;
        z_pos_h[i] = 0.0;
        r_h[i] = 0.1;
    }
    colSystem->update_all_from_host();
    initKernel<<<1,1>>>(colSystem_d);
    find_cols_tree_Kernel<<<1,1>>>(colSystem_d);
    cudaMemcpy((void*)&numbCollisionsTree, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();


    find_cols_N2_Kernel<<<1,1>>>(colSystem_d);
    cudaMemcpy((void*)&numbCollisionsN2, colSystem->num_collisions_d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();


    delete colSystem; // keep the colSystem_d -- we will reuse the memory.
    cudaFree(colSystem_d);
    if (numbCollisionsN2 != 3) {
        if (verbose > 0) {
            std::cout << "N2 algorithm failed 3 particle test (B). Expected 3 collision. Found: " << numbCollisionsN2 << " collisions." << std::endl;
        }
        return false;
    }
    if (numbCollisionsTree != 3) {
        if (verbose > 0) {
            std::cout << "Tree algorithm failed 3 particle test (B). Expected 3 collision. Found: " << numbCollisionsTree << " collisions." << std::endl;
        }
        return false;
    }
    return true;
}

bool testN2CollisionSystemHeuristic(int verbose) {
    thrust::default_random_engine rng(0);
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    unsigned int N = 1<<10;
    CollisionSystem *colSystem = new CollisionSystem(N, N, true);


    auto x_pos_h = colSystem->get_x_pos_host();
    auto y_pos_h = colSystem->get_y_pos_host();
    auto z_pos_h = colSystem->get_z_pos_host();
    auto r_h = colSystem->get_radius_host();

    // init positions
    for (int i = 0; i < colSystem->get_num_masses(); i++) {
        x_pos_h[i] = u01(rng);
        y_pos_h[i] = u01(rng);
        z_pos_h[i] = u01(rng);
        r_h[i] = u01(rng);
    }

    colSystem->update_all_from_host();

    unsigned int numColsGPU = colSystem->find_collisions_N2();
    thrust::device_ptr<Collision> col_d_ptr(colSystem->collisions_d_ptr);
    thrust::sort(col_d_ptr, col_d_ptr + numColsGPU);

    thrust::host_vector<Collision> collisions_cpu_h(
            colSystem->get_num_masses() * colSystem->get_max_collisions_per_mass());
    thrust::device_vector<Collision> collisions_cpu_d(
            colSystem->get_num_masses() * colSystem->get_max_collisions_per_mass());
    thrust::counting_iterator<unsigned int> start(0);

    unsigned int numColsCPU = find_collisions_cpu(colSystem->get_num_masses(),
            x_pos_h,
            y_pos_h,
            z_pos_h,
            r_h,
            thrust::raw_pointer_cast(collisions_cpu_h.data()));

    collisions_cpu_d = collisions_cpu_h;
    thrust::sort(collisions_cpu_d.begin(), collisions_cpu_d.begin() + numColsCPU);
    if (verbose > 0) {
        std::cout << std::setw(10) << numColsGPU << " GPU collisions found ... " << std::flush;
    }

    bool error = false;
    if (numColsGPU != numColsCPU) {
        if (verbose > 0) {
            std::cout << numColsCPU << " C, G " << numColsGPU << " ERROR!" << std::endl;
            std::cout << "GPU and CPU found different numbers of collisions!" << std::endl;
        }
        error = true;
    }

    if (!thrust::equal(collisions_cpu_d.begin(), collisions_cpu_d.begin() + numColsCPU, col_d_ptr)){
        if (verbose > 0){
            std::cout << "Error!\nGPU and CPU collision detection diverged!" << std::endl;

        }
        error = true;
    }
    if (verbose > 0) {
        std::cout << " passed" << std::flush;
    }

    delete colSystem;

    return !error;
}


bool testTreeCollisionSystemHeuristic(int verbose) {
    thrust::default_random_engine rng(0);
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    thrust::uniform_real_distribution<float> u11(-1.f, 1.f);
    CollisionSystem *colSystem = nullptr;

    for (int loops = 0; loops < MAX_LOOPS; loops++) {
        for (int levelsIdx = 0; levelsIdx < sizeof(levels) / sizeof(unsigned int); levelsIdx++) {
            unsigned int N = levels[levelsIdx];
            if (verbose > 0) {
                std::cout << "\rLoop " << std::setw(ceil(log10(MAX_LOOPS))) << loops << " of " << MAX_LOOPS
                          << " running on " << std::setw(8) << N << " points ... " << std::flush;
            }

            if (colSystem != nullptr) {
                delete colSystem;
            }

            if (MAX_COLS_PER_OBJECT > 0) {
                colSystem = new CollisionSystem(N, MAX_COLS_PER_OBJECT, true);
            } else {
                colSystem = new CollisionSystem(N, N, true);
            }

            auto x_pos_h = colSystem->get_x_pos_host();
            auto y_pos_h = colSystem->get_y_pos_host();
            auto z_pos_h = colSystem->get_z_pos_host();
            auto r_h = colSystem->get_radius_host();

            // init positions
            for (int i = 0; i < colSystem->get_num_masses(); i++) {
                x_pos_h[i] = u11(rng);
                y_pos_h[i] = u11(rng);
                z_pos_h[i] = u11(rng);
                r_h[i] = u01(rng);
            }

            colSystem->update_all_from_host();
            colSystem->update_x_pos_ranks();
            colSystem->update_y_pos_ranks();
            colSystem->update_z_pos_ranks();
            colSystem->update_mortons();
            colSystem->build_tree();
            colSystem->update_bounding_boxes();

            unsigned int numColsGPU = colSystem->find_collisions();
            thrust::device_ptr<Collision> col_d_ptr(colSystem->collisions_d_ptr);
            thrust::sort(col_d_ptr, col_d_ptr + numColsGPU);


            thrust::host_vector<Collision> collisions_cpu_h(
                    colSystem->get_num_masses() * colSystem->get_max_collisions_per_mass());
            thrust::device_vector<Collision> collisions_cpu_d(
                    colSystem->get_num_masses() * colSystem->get_max_collisions_per_mass());
            thrust::counting_iterator<unsigned int> start(0);

            unsigned int numColsCPU = find_collisions_cpu(colSystem->get_num_masses(),
                    x_pos_h,
                    y_pos_h,
                    z_pos_h,
                    r_h,
                    thrust::raw_pointer_cast(collisions_cpu_h.data()));

            collisions_cpu_d = collisions_cpu_h;
            thrust::sort(collisions_cpu_d.begin(), collisions_cpu_d.begin() + numColsCPU);
            if (verbose > 0) {
                std::cout << std::setw(10) << numColsGPU << " GPU collisions found ... " << std::flush;
            }

            bool error = false;
            if (numColsGPU != numColsCPU) {
                if (verbose > 0) {
                    std::cout << numColsCPU << " C, G " << numColsGPU << " ERROR!" << std::endl;
                    std::cout << "GPU and CPU found different numbers of collisions!" << std::endl;
                }
                error = true;
            }

            if (!thrust::equal(thrust::device, collisions_cpu_d.begin(), collisions_cpu_d.begin() + numColsCPU, col_d_ptr)) {
                if (verbose > 0){
                    std::cout << "Error!\nGPU and CPU collision detection diverged!" << std::endl;

                }
                error = true;
            }

            if (error && verbose > 1) {
                thrust::host_vector<Collision> collisions_cpu_h = collisions_cpu_d;
                thrust::host_vector<Collision> collisions_gpu_h(col_d_ptr, col_d_ptr + numColsGPU );

                unsigned int cpuIdx = 0;
                unsigned int gpuIdx = 0;
                while (cpuIdx < numColsCPU || gpuIdx < numColsGPU) {
                    if (cpuIdx >= numColsCPU) {
                        std::cout << "EndIdx Only on GPU: " << collisions_gpu_h[gpuIdx] << std::endl;
                        unsigned int a, b;
                        a = collisions_gpu_h[gpuIdx].a;
                        b = collisions_gpu_h[gpuIdx].b;
                        std::cout << "a (r: " << r_h[a] << "), b (r: " << r_h[b] << ") -- (" << x_pos_h[a] << ", " << y_pos_h[a] << ", " << z_pos_h[a] << ") ("
                                  << x_pos_h[b] << ", " << y_pos_h[b] << ", " << z_pos_h[b] << ") " << std::endl;
                        gpuIdx++;
                    } else if (gpuIdx >= numColsGPU) {
                        std::cout << "EndIdx Only on CPU: " << collisions_cpu_h[cpuIdx] << std::endl;
                        unsigned int a, b;
                        a = collisions_cpu_h[gpuIdx].a;
                        b = collisions_cpu_h[gpuIdx].b;
                        std::cout << "a (r: " << r_h[a] << "), b (r: " << r_h[b] << ") -- (" << x_pos_h[a] << ", " << y_pos_h[a] << ", " << z_pos_h[a] << ") ("
                                  << x_pos_h[b] << ", " << y_pos_h[b] << ", " << z_pos_h[b] << ")" << std::endl;
                        cpuIdx++;
                    } else {
                        if (collisions_cpu_h[cpuIdx] == collisions_gpu_h[gpuIdx]) {
                            cpuIdx++;
                            gpuIdx++;
                        } else if (collisions_cpu_h[cpuIdx] < collisions_gpu_h[gpuIdx]) {
                            std::cout << "MiddleIdx Only on CPU: " << collisions_cpu_h[cpuIdx] << std::endl;
                            unsigned int a, b;
                            a = collisions_cpu_h[gpuIdx].a;
                            b = collisions_cpu_h[gpuIdx].b;
                            std::cout << "a (r: " << r_h[a] << "), b (r: " << r_h[b] << ") -- (" << x_pos_h[a] << ", " << y_pos_h[a] << ", " << z_pos_h[a] << ") ("
                                      << x_pos_h[b] << ", " << y_pos_h[b] << ", " << z_pos_h[b] << ")" << std::endl;
                            cpuIdx++;
                        } else {
                            std::cout << "MiddleIdx Only on GPU: " << collisions_gpu_h[gpuIdx] << std::endl;
                            unsigned int a, b;
                            a = collisions_gpu_h[gpuIdx].a;
                            b = collisions_gpu_h[gpuIdx].b;
                            std::cout << "a (r: " << r_h[a] << "), b (r: " << r_h[b] << ") -- ("  << x_pos_h[a] << ", " << y_pos_h[a] << ", " << z_pos_h[a] << ") ("
                                      << x_pos_h[b] << ", " << y_pos_h[b] << ", " << z_pos_h[b] << ")" << std::endl;
                            gpuIdx++;
                        }
                    }
                }
            }

            if (error) {
                return false;
            }
            if (verbose > 0) {
                std::cout << " passed" << std::flush;
            }
        }
    }
    return true;
}

