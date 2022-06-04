//
// Created by David Matthews on 5/21/20.
//


#include "../include/CollisionSystem.cuh"

CollisionSystem::CollisionSystem(size_t n, size_t maxCollisionsPerObject, bool toAllocHostVecs) : N(n), MAX_COLLISIONS_PER_OBJECT(maxCollisionsPerObject) {

    start = thrust::counting_iterator<unsigned int>(0);
    // allocate memory for positions
    // host
    allocatedByUs = toAllocHostVecs;

    RESERVATION_SIZE = N;

    cudaMalloc((void**)&num_collisions_d_ptr, 1 * sizeof(int));

    // CUB sort buffer
    cub_sort_bytes_size = 1; // enlarge upon request in kernels.
    cudaMalloc(&cub_sort_bytes_ptr, cub_sort_bytes_size);
    CUDA_CHECK_AFTER_CALL();

    set_num_objects(n);

}

CollisionSystem::~CollisionSystem() {
    if (allocatedByUs) {
        delete[] x_pos_h;
        delete[] y_pos_h;
        delete[] z_pos_h;
        delete[] radius_h;
        delete[] host_collisions_a;
    }
    cudaFree(cub_sort_bytes_ptr);

    cudaFree(x_pos_d_ptr);
    cudaFree(y_pos_d_ptr);
    cudaFree(z_pos_d_ptr);
    cudaFree(radius_d_ptr);
    cudaFree(tmp_pos_d_ptr);
    cudaFree(x_rank_d_ptr);
    cudaFree(y_rank_d_ptr);
    cudaFree(z_rank_d_ptr);
    cudaFree(tmp_id_a_d_ptr);
    cudaFree(tmp_id_b_d_ptr);
    cudaFree(mortons_d_ptr);
    cudaFree(mortons_tmp_d_ptr);
    cudaFree(mortons_id_d_ptr);
    cudaFree(leaf_parent_d_ptr);
    cudaFree(internal_parent_d_ptr);
    cudaFree(internal_childA_d_ptr);
    cudaFree(internal_childB_d_ptr);
    cudaFree(internal_node_bbox_complete_flag_d_ptr);
    cudaFree(bounding_boxes_d_ptr);
    cudaFree(potential_collisions_idx_d_ptr);
    cudaFree(potential_collisions_d_ptr);
    cudaFree(collisions_d_ptr);
}

void CollisionSystem::set_num_objects_host(size_t n) {

    if (allocatedByUs) {
        // avoid need for special case by ensuring host data is not null (only occurs when first constructing)
        if (!x_pos_h) {  x_pos_h = new float[RESERVATION_SIZE];  }
        if (!y_pos_h) { y_pos_h = new float[RESERVATION_SIZE]; }
        if (!z_pos_h) { z_pos_h = new float[RESERVATION_SIZE]; }
        if (!radius_h) { radius_h = new float[RESERVATION_SIZE]; }
        if (!host_collisions_a) { host_collisions_a = new Collision[RESERVATION_SIZE * MAX_COLLISIONS_PER_OBJECT]; }

        float *tmp_x, *tmp_y, *tmp_z, *tmp_r;

        tmp_x = new float[RESERVATION_SIZE];
        thrust::copy(x_pos_h, x_pos_h + N, tmp_x);
        delete[] x_pos_h;
        x_pos_h = tmp_x;

        tmp_y = new float[RESERVATION_SIZE];
        thrust::copy(y_pos_h, y_pos_h + N, tmp_y);
        delete[] y_pos_h;
        y_pos_h = tmp_y;

        tmp_z = new float[RESERVATION_SIZE];
        thrust::copy(z_pos_h, z_pos_h + N, tmp_z);
        delete[] z_pos_h;
        z_pos_h = tmp_z;

        tmp_r = new float[RESERVATION_SIZE];
        thrust::copy(radius_h, radius_h + N, tmp_r);
        delete[] radius_h;
        radius_h = tmp_r;

        auto *tmp_h = new Collision[RESERVATION_SIZE * MAX_COLLISIONS_PER_OBJECT];
        thrust::copy(host_collisions_a, host_collisions_a + N, tmp_h);
        delete[] host_collisions_a;
        host_collisions_a = tmp_h;
    }
}

__host__ __device__
void CollisionSystem::set_num_objects_device(size_t n) {
    if (N != n) {
        requiresRebuild = true;
    }

    N = n;
    if (N > RESERVATION_SIZE) {
        RESERVATION_SIZE = N;
    } else if (!needAllocate) {
       update_device_pointers_and_functors();
       return;
    }

    // device
    cudaFree(x_pos_d_ptr);
    cudaMalloc((void**)&x_pos_d_ptr, RESERVATION_SIZE * sizeof(float));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(y_pos_d_ptr);
    cudaMalloc((void**)&y_pos_d_ptr, RESERVATION_SIZE * sizeof(float));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(z_pos_d_ptr);
    cudaMalloc((void**)&z_pos_d_ptr, RESERVATION_SIZE * sizeof(float));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(radius_d_ptr);
    cudaMalloc((void**)&radius_d_ptr, RESERVATION_SIZE * sizeof(float));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(tmp_pos_d_ptr);
    cudaMalloc((void**)&tmp_pos_d_ptr, RESERVATION_SIZE * sizeof(float));
    CUDA_CHECK_AFTER_CALL();

    // alloc rank and id vectors.
    cudaFree(x_rank_d_ptr);
    cudaMalloc((void**)&x_rank_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(y_rank_d_ptr);
    cudaMalloc((void**)&y_rank_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(z_rank_d_ptr);
    cudaMalloc((void**)&z_rank_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(tmp_id_a_d_ptr);
    cudaMalloc((void**)&tmp_id_a_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(tmp_id_b_d_ptr);
    cudaMalloc((void**)&tmp_id_b_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    // allocate Morton number vectors
    cudaFree(mortons_d_ptr);
    cudaMalloc((void**)&mortons_d_ptr, RESERVATION_SIZE * sizeof(unsigned long long int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(mortons_tmp_d_ptr);
    cudaMalloc((void**)&mortons_tmp_d_ptr, RESERVATION_SIZE * sizeof(unsigned long long int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(mortons_id_d_ptr);
    cudaMalloc((void**)&mortons_id_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    // alloc vectors for the BVH tree
    // for the leaf nodes
    cudaFree(leaf_parent_d_ptr);
    cudaMalloc((void**)&leaf_parent_d_ptr, RESERVATION_SIZE * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    // for the internal nodes
    cudaFree(internal_parent_d_ptr);
    cudaMalloc((void**)&internal_parent_d_ptr, (RESERVATION_SIZE - 1) * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(internal_childA_d_ptr);
    cudaMalloc((void**)&internal_childA_d_ptr, (RESERVATION_SIZE - 1) * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(internal_childB_d_ptr);
    cudaMalloc((void**)&internal_childB_d_ptr, (RESERVATION_SIZE - 1) * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    // for the bounding boxes for all leaf and internal nodes.
    cudaFree(internal_node_bbox_complete_flag_d_ptr);
    cudaMalloc((void**)&internal_node_bbox_complete_flag_d_ptr, (RESERVATION_SIZE - 1) * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(bounding_boxes_d_ptr);
    cudaMalloc((void**)&bounding_boxes_d_ptr, (2 * RESERVATION_SIZE - 1) * sizeof(BoundingBox));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(potential_collisions_idx_d_ptr);
    cudaMalloc((void**)&potential_collisions_idx_d_ptr, (1) * sizeof(unsigned int));
    CUDA_CHECK_AFTER_CALL();

    assert(MAX_COLLISIONS_PER_OBJECT * N != 0); // "Size of potential_collisions array must be > 0";

    // TODO: can we automatically expand collision memory as needed?

    cudaFree(potential_collisions_d_ptr);
    cudaMalloc((void**)&potential_collisions_d_ptr, (MAX_COLLISIONS_PER_OBJECT * RESERVATION_SIZE) * sizeof(Collision));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(collisions_d_ptr);
    cudaMalloc((void**)&collisions_d_ptr, (MAX_COLLISIONS_PER_OBJECT * RESERVATION_SIZE) * sizeof(Collision));
    CUDA_CHECK_AFTER_CALL();

    update_device_pointers_and_functors();

    // init flags to zero.
    thrust::fill(thrust::device, internal_node_bbox_complete_flag_d_ptr, internal_node_bbox_complete_flag_d_ptr + N - 1, 0);
    needAllocate = false;

}

void CollisionSystem::set_max_num_cols_per_mass(size_t m) {
    MAX_COLLISIONS_PER_OBJECT = m;

    cudaFree(potential_collisions_d_ptr);
    cudaMalloc((void**)&potential_collisions_d_ptr, (MAX_COLLISIONS_PER_OBJECT * RESERVATION_SIZE) * sizeof(Collision));
    CUDA_CHECK_AFTER_CALL();

    cudaFree(collisions_d_ptr);
    cudaMalloc((void**)&collisions_d_ptr, (MAX_COLLISIONS_PER_OBJECT * RESERVATION_SIZE) * sizeof(Collision));
    CUDA_CHECK_AFTER_CALL();

    update_device_pointers_and_functors();
}

__host__ __device__
void CollisionSystem::update_device_pointers_and_functors() {
    compute_morton_numbers = init_morton_func(x_rank_d_ptr,
            y_rank_d_ptr,
            z_rank_d_ptr,
            mortons_d_ptr,
            mortons_id_d_ptr);

    build_bvh_tree = build_bvh_tree_func(N,
            mortons_d_ptr,
            leaf_parent_d_ptr,
            internal_parent_d_ptr,
            internal_childA_d_ptr,
            internal_childB_d_ptr);

    compute_bounding_boxes = fill_bvh_tree_with_bounding_boxes_func(N,
            bounding_boxes_d_ptr,
            x_pos_d_ptr,
            y_pos_d_ptr,
            z_pos_d_ptr,
            radius_d_ptr,
            mortons_id_d_ptr,
            leaf_parent_d_ptr,
            internal_parent_d_ptr,
            internal_childA_d_ptr,
            internal_childB_d_ptr,
            internal_node_bbox_complete_flag_d_ptr);

    find_potential_collisions = find_potential_collisions_func(N,
            N * MAX_COLLISIONS_PER_OBJECT,
            mortons_id_d_ptr, bounding_boxes_d_ptr, internal_childA_d_ptr, internal_childB_d_ptr, potential_collisions_idx_d_ptr,
            potential_collisions_d_ptr, x_pos_d_ptr, y_pos_d_ptr, z_pos_d_ptr, radius_d_ptr);

    check_potential_collisions = check_potential_collisions_func(x_pos_d_ptr, y_pos_d_ptr, z_pos_d_ptr, radius_d_ptr);
}

void CollisionSystem::update_all_from_host() {
    update_x_pos_from_host();
    update_y_pos_from_host();
    update_z_pos_from_host();
    update_radius_from_host();
}

void CollisionSystem::update_x_pos_from_host() {
    cudaMemcpy(x_pos_d_ptr, x_pos_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();
}

void CollisionSystem::update_y_pos_from_host() {
    cudaMemcpy(y_pos_d_ptr, y_pos_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();}

void CollisionSystem::update_z_pos_from_host() {
    cudaMemcpy(z_pos_d_ptr, z_pos_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();}

void CollisionSystem::update_radius_from_host() {
    cudaMemcpy(radius_d_ptr, radius_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();}

void CollisionSystem::init() {
    // copy from host to device
    update_all_from_host();

    // compute ranks
    update_x_pos_ranks();
    update_y_pos_ranks();
    update_z_pos_ranks();

    // build and sort mortons
    update_mortons();

    // build BVH tree
    build_tree();

    // fill BVH tree with bounding boxes
    update_bounding_boxes();
}

__host__ __device__
void CollisionSystem::update_x_pos_ranks() {
    // keep track of x object ids after sorting.
    thrust::sequence(thrust::device, tmp_id_a_d_ptr, tmp_id_a_d_ptr + N);

    size_t curr_size;
    cub::DeviceRadixSort::SortPairs(NULL, curr_size, x_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);
    if (curr_size > cub_sort_bytes_size) {
        cub_sort_bytes_size = curr_size;
        cudaFree(cub_sort_bytes_ptr);
        cudaMalloc(&cub_sort_bytes_ptr, cub_sort_bytes_size);
    }
    cub::DeviceRadixSort::SortPairs(cub_sort_bytes_ptr, curr_size, x_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);

    // save the new rank information
    thrust::scatter(thrust::device, start, start + N, tmp_id_b_d_ptr, x_rank_d_ptr);
}

__host__ __device__
void CollisionSystem::update_y_pos_ranks() {
    // keep track of y object ids after sorting
    thrust::sequence(thrust::device, tmp_id_a_d_ptr, tmp_id_a_d_ptr + N);

    // sort the positions to determine rank.
    size_t curr_size;
    cub::DeviceRadixSort::SortPairs(NULL, curr_size, y_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);
    if (curr_size > cub_sort_bytes_size) {
        cub_sort_bytes_size = curr_size;
        cudaFree(cub_sort_bytes_ptr);
        cudaMalloc(&cub_sort_bytes_ptr, cub_sort_bytes_size);
    }
    cub::DeviceRadixSort::SortPairs(cub_sort_bytes_ptr, curr_size, y_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);

    // save the new rank information
    thrust::scatter(thrust::device, start, start + N, tmp_id_b_d_ptr, y_rank_d_ptr);
}

__host__ __device__
void CollisionSystem::update_z_pos_ranks() {
    // keep track of z object ids after sorting
    thrust::sequence(thrust::device, tmp_id_a_d_ptr, tmp_id_a_d_ptr + N);


    // sort the positions to determine rank.
    size_t curr_size;
    cub::DeviceRadixSort::SortPairs(NULL, curr_size, z_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);
    if (curr_size > cub_sort_bytes_size) {
        cub_sort_bytes_size = curr_size;
        cudaFree(cub_sort_bytes_ptr);
        cudaMalloc(&cub_sort_bytes_ptr, cub_sort_bytes_size);
    }
    cub::DeviceRadixSort::SortPairs(cub_sort_bytes_ptr, curr_size, z_pos_d_ptr, tmp_pos_d_ptr, tmp_id_a_d_ptr, tmp_id_b_d_ptr, N);

    // save the new rank information
    thrust::scatter(thrust::device, start, start + N, tmp_id_b_d_ptr, z_rank_d_ptr);
}

__host__ __device__
void CollisionSystem::update_mortons() {
    // keep track of object ids after sorting.
    thrust::sequence(thrust::device, tmp_id_a_d_ptr, tmp_id_a_d_ptr + N);

    // build morton numbers.
    thrust::for_each(thrust::device, start, start + N, compute_morton_numbers);
    thrust::copy(thrust::device, mortons_d_ptr, mortons_d_ptr + N, mortons_tmp_d_ptr); // copy mortons to tmp array as source for sorting.


    // sort morton numbers
    size_t curr_size;
    cub::DeviceRadixSort::SortPairs(NULL, curr_size, mortons_tmp_d_ptr, mortons_d_ptr, tmp_id_a_d_ptr, mortons_id_d_ptr, N);
    if (curr_size > cub_sort_bytes_size) {
        cub_sort_bytes_size = curr_size;
        cudaFree(cub_sort_bytes_ptr);
        cudaMalloc(&cub_sort_bytes_ptr, cub_sort_bytes_size);
    }
    cub::DeviceRadixSort::SortPairs(cub_sort_bytes_ptr, curr_size, mortons_tmp_d_ptr, mortons_d_ptr, tmp_id_a_d_ptr, mortons_id_d_ptr, N);
}

__host__ __device__
void CollisionSystem::update_mortons_fast(float2 xlims, float2 ylims, float2 zlims) {
    thrust::sequence(thrust::device, mortons_id_d_ptr, mortons_id_d_ptr + N);

    // build morton numbers using the Karras method.
    // this will be faster if we are not simulating swarms of particles (e.g. if voxels are evenly distributed across range)
    // but slower if we are simulating swarms of particles that encompass large amounts of area and are not evenly distributed
    // e.g. voxels clumping to form new robots would likely be slower with this method.

    thrust::for_each(thrust::device, start,
            start + N,
            init_morton_func_fast(xlims,
                    ylims,
                    zlims,
                    x_pos_d_ptr, y_pos_d_ptr, z_pos_d_ptr, mortons_d_ptr, mortons_id_d_ptr));

    // sort morton numbers
    thrust::sort_by_key(thrust::device, mortons_d_ptr, mortons_d_ptr + N, mortons_id_d_ptr);
}

__host__ __device__
void CollisionSystem::build_tree() {
    build_bvh_tree.N = N;
    thrust::for_each(thrust::device, start, start + N - 1 , build_bvh_tree);

    // int num_SM, curDeviceId, gridSize, blockSize;
    // cudaGetDevice(&curDeviceId);
    // cudaDeviceGetAttribute(&num_SM, cudaDevAttrMultiProcessorCount, curDeviceId);
    // blockSize = (int)(N-1)/num_SM;
    // if (num_SM * blockSize < (N-1)) {
    //     blockSize += 1;
    // }
    // if (blockSize > 256) {
    //     blockSize = 256;
    //     gridSize = ((int)N + 254)/256; // N - 1 + 255 leaf nodes.
    // } else {
    //     gridSize = num_SM;
    // }
    // build_tree_kernel<<<gridSize, blockSize>>>(0, N - 1, build_bvh_tree);
    // CUDA_CHECK_AFTER_CALL();
    // VcudaDeviceSynchronize();
}

__host__ __device__
void CollisionSystem::update_bounding_boxes() {
    compute_bounding_boxes.N = N;
    thrust::for_each(thrust::device, start, start + N, compute_bounding_boxes);
}

__host__
bool CollisionSystem::check_collisions(float pX, float pY, float pZ, float pR) {
    thrust::device_vector<bool> result(1);
    check_collisions_single<<<1,1>>>(find_potential_collisions, pX, pY, pZ, pR, thrust::raw_pointer_cast(result.data()));
    return result[0];
}

__device__
bool CollisionSystem::check_collisions_device(float pX, float pY, float pZ, float pR) {
    return find_potential_collisions.test_collision(pX, pY, pZ, pR);
}

__host__
int CollisionSystem::find_collisions() {
    cudaMemset(potential_collisions_idx_d_ptr, 0, sizeof(unsigned int));
    find_potential_collisions.N = N;
    find_potential_collisions.NUM_INTERNAL = N - 1;
    thrust::for_each(thrust::device, start + N - 1, start + 2 * N - 1, find_potential_collisions);

    unsigned int h_potential_collision_idx;
    cudaMemcpy((void*)&h_potential_collision_idx, (void*)potential_collisions_idx_d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (h_potential_collision_idx > MAX_COLLISIONS_PER_OBJECT * N) {
        cudaMemset(num_collisions_d_ptr, -1, sizeof(int));
        return -1;
    }

    unsigned int colCount = thrust::copy_if(thrust::device, potential_collisions_d_ptr,
            potential_collisions_d_ptr + h_potential_collision_idx,
            collisions_d_ptr,
            check_potential_collisions) - collisions_d_ptr;

    cudaMemset(num_collisions_d_ptr, colCount, sizeof(int));
    return colCount;
}

__device__
int CollisionSystem::find_collisions_device(int pruneLevel) {
    potential_collisions_idx_d_ptr[0] = 0;
    find_potential_collisions.N = N;
    find_potential_collisions.NUM_INTERNAL = N - 1;

    int num_SM, curDeviceId, gridSize, blockSize;
    cudaGetDevice(&curDeviceId);
    cudaDeviceGetAttribute(&num_SM, cudaDevAttrMultiProcessorCount, curDeviceId);
    blockSize = (int)N/num_SM;
    if (num_SM * blockSize < N) {
        blockSize += 1;
    }
    if (blockSize > 256) {
        blockSize = 256;
        gridSize = ((int)N + 255)/256;
    } else {
        gridSize = num_SM;
    }
    find_potential_collisions_kernel<<<gridSize, blockSize>>>(N - 1, N, find_potential_collisions);
    CUDA_CHECK_AFTER_CALL();
    VcudaDeviceSynchronize();

    if (potential_collisions_idx_d_ptr[0] > MAX_COLLISIONS_PER_OBJECT * N) {
        num_collisions_d_ptr[0] = -1;
        return -1;
    }
    if (pruneLevel == 0) {
        num_collisions_d_ptr[0] = (int) potential_collisions_idx_d_ptr[0];
        thrust::copy(thrust::device, potential_collisions_d_ptr,potential_collisions_d_ptr +   potential_collisions_idx_d_ptr[0], collisions_d_ptr);
    } else {
        unsigned int colCount = thrust::copy_if(thrust::device, potential_collisions_d_ptr,
                potential_collisions_d_ptr + potential_collisions_idx_d_ptr[0],
                collisions_d_ptr,
                check_potential_collisions) - collisions_d_ptr;
        num_collisions_d_ptr[0] = (int) colCount;
    }
    return num_collisions_d_ptr[0];
}

__host__
int CollisionSystem::find_collisions_N2() {
    auto keys_a_start = thrust::make_transform_iterator(start, thrust::placeholders::_1 / N);
    auto keys_b_start = thrust::make_transform_iterator(start, thrust::placeholders::_1 % N);

    auto keys_zip_start = thrust::make_zip_iterator(thrust::make_tuple(keys_a_start, keys_b_start));
    auto keys_zip_end = thrust::make_zip_iterator(thrust::make_tuple(keys_a_start + N * N, keys_b_start + N * N));

    cudaMemset(potential_collisions_idx_d_ptr, 0, sizeof(unsigned int));
    thrust::for_each(thrust::device, keys_zip_start,
            keys_zip_end,
            check_potential_collisions_N2_func(N * MAX_COLLISIONS_PER_OBJECT,
                    potential_collisions_idx_d_ptr, x_pos_d_ptr, y_pos_d_ptr, z_pos_d_ptr, radius_d_ptr, collisions_d_ptr));

    unsigned int num_collisions_h;
    cudaMemcpy((void*)&num_collisions_h, (void*)potential_collisions_idx_d_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (num_collisions_h > MAX_COLLISIONS_PER_OBJECT * N) {
        cudaMemset(num_collisions_d_ptr, -1, sizeof(int));
        return -1;
    }

    cudaMemset(num_collisions_d_ptr, num_collisions_h, sizeof(int));
    return num_collisions_h;
}

__device__
int CollisionSystem::find_collisions_N2_device() {
    auto keys_a_start = thrust::make_transform_iterator(start, thrust::placeholders::_1 / N);
    auto keys_b_start = thrust::make_transform_iterator(start, thrust::placeholders::_1 % N);

    auto keys_zip_start = thrust::make_zip_iterator(thrust::make_tuple(keys_a_start, keys_b_start));
    auto keys_zip_end = thrust::make_zip_iterator(thrust::make_tuple(keys_a_start + N * N, keys_b_start + N * N));

    potential_collisions_idx_d_ptr[0] = 0;
    thrust::for_each(thrust::device, keys_zip_start,
            keys_zip_end,
            check_potential_collisions_N2_func(N * MAX_COLLISIONS_PER_OBJECT,
                    potential_collisions_idx_d_ptr, x_pos_d_ptr, y_pos_d_ptr, z_pos_d_ptr, radius_d_ptr, collisions_d_ptr));

    if (potential_collisions_idx_d_ptr[0] > MAX_COLLISIONS_PER_OBJECT * N) {
        return -1;
    }

    num_collisions_d_ptr[0] = potential_collisions_idx_d_ptr[0];
    return potential_collisions_idx_d_ptr[0];
}

__global__ void check_collisions_single(find_potential_collisions_func functor, float pX, float pY, float pZ, float pR, bool *b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        *b = functor.test_collision(pX, pY, pZ, pR);
    }
}

__global__ void find_potential_collisions_kernel(int startIdx, int num, find_potential_collisions_func functor) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        functor(tid + startIdx);
    }
}
__global__ void build_tree_kernel(int startIdx, int num, build_bvh_tree_func functor) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        functor(tid + startIdx);
    }
}