//
// Created by David Matthews on 5/23/20.
//
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/CollisionSystem.cuh"

namespace py = pybind11;

PYBIND11_MODULE(pycol, m) {
    py::class_<CollisionSystem>(m, "CollisionSystem")
            .def(py::init<>())
            .def(py::init([](int N, int max_cols_per_mass) {
                auto colSys = new CollisionSystem(N, max_cols_per_mass, false);
                return colSys;
            }))
            .def(py::init([](size_t N, size_t max_cols_per_mass,
                                                                      py::array_t<float> pos,
                                                                      py::array_t<int> collisions)
                                                                      {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                py::buffer_info colsInfo = collisions.request();
                if (colsInfo.format != py::format_descriptor<int>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! colsInfo.format != int" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
//                if (colsInfo.ndim != 2) {
//                    std::stringstream strstr;
//                    strstr << "Incompatible buffer format! colsInfo.ndim != 2" << std::endl;
//                    strstr << "colsInfo.ndim: " << colsInfo.ndim << std::endl;
//                    throw std::runtime_error(strstr.str());
//                }

                auto colSys = new CollisionSystem(N, max_cols_per_mass, false);
                colSys->set_x_pos_host((float *) posInfo.ptr);
                colSys->set_y_pos_host((float *) posInfo.ptr + posInfo.strides[0]/sizeof(float) * 1);
                colSys->set_z_pos_host((float *) posInfo.ptr + posInfo.strides[0]/sizeof(float) * 2);
                colSys->set_radius_host((float *) posInfo.ptr + posInfo.strides[0]/sizeof(float) * 3);
                colSys->set_collisions_host((Collision *) colsInfo.ptr);
                return colSys;
            }))
            .def("set_pos_host", [](CollisionSystem &colSys, py::array_t<float> pos) {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_x_pos_host((float *) posInfo.ptr);
                colSys.set_y_pos_host((float *) posInfo.ptr + posInfo.strides[0] * 1);
                colSys.set_z_pos_host((float *) posInfo.ptr + posInfo.strides[0] * 2);
                colSys.set_radius_host((float *) posInfo.ptr + posInfo.strides[0] * 3);
            })
            .def("set_x_pos_host", [](CollisionSystem &colSys, py::array_t<float> pos) {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_x_pos_host((float *) posInfo.ptr);
            })
            .def("set_y_pos_host", [](CollisionSystem &colSys, py::array_t<float> pos) {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_y_pos_host((float *) posInfo.ptr);
            })
            .def("set_z_pos_host", [](CollisionSystem &colSys, py::array_t<float> pos) {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_z_pos_host((float *) posInfo.ptr);
            })
            .def("set_radius_host", [](CollisionSystem &colSys, py::array_t<float> pos) {
                py::buffer_info posInfo = pos.request();

                if (posInfo.format != py::format_descriptor<float>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.format != float32" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (posInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! posInfo.ndim != 2" << std::endl;
                    strstr << "posInfo.ndim: " << posInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_radius_host((float *) posInfo.ptr);
            })
            .def("set_collisions_host", [](CollisionSystem &colSys, py::array_t<int> collisions) {
                py::buffer_info colsInfo = collisions.request();
                if (colsInfo.format != py::format_descriptor<int>::format()) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! colsInfo.format != int" << std::endl;
                    throw std::runtime_error(strstr.str());
                }
                if (colsInfo.ndim != 2) {
                    std::stringstream strstr;
                    strstr << "Incompatible buffer format! colsInfo.ndim != 2" << std::endl;
                    strstr << "colsInfo.ndim: " << colsInfo.ndim << std::endl;
                    throw std::runtime_error(strstr.str());
                }

                colSys.set_collisions_host((Collision *) colsInfo.ptr);
            })
            .def("set_collisions_host", &CollisionSystem::set_collisions_host)
            .def("__len__", &CollisionSystem::get_num_masses, "Returns the number of masses in the system")
            .def("get_num_masses", &CollisionSystem::get_num_masses, "Returns the number of masses in the system")
            .def("get_max_collisions_per_object",
                    &CollisionSystem::get_max_collisions_per_mass,
                    "Returns the maximum number of collisions per object supported by the system")
            .def("set_num_masses",
                    &CollisionSystem::set_num_masses,
                    "Sets the maximum number masses supported by the system")
            .def("set_reserved_num_masses",
                    &CollisionSystem::set_reserved_num_masses,
                    "Reserves more memory for the system")
            .def("set_max_num_cols_per_mass", &CollisionSystem::set_max_num_cols_per_mass)
            .def("update_all_from_host",
                    &CollisionSystem::update_all_from_host,
                    "Copies position and radius data from host vectors to device vectors")
            .def("update_x_pos_from_host",
                    &CollisionSystem::update_x_pos_from_host,
                    "Copies x pos data from host to device vectors")
            .def("update_y_pos_from_host",
                    &CollisionSystem::update_y_pos_from_host,
                    "Copies y pos data from host to device vectors")
            .def("update_z_pos_from_host",
                    &CollisionSystem::update_z_pos_from_host,
                    "Copies z pos data from host to device vectors")
            .def("update_radius_from_host",
                    &CollisionSystem::update_radius_from_host,
                    "Copies radius data from host to device vectors")
            .def("init",
                    &CollisionSystem::init,
                    "helper method to copy all data from host to device, then build ranked morton numbers then build and populate the BVH tree")
            .def("update_x_pos_ranks",
                    &CollisionSystem::update_x_pos_ranks,
                    "Computes the rank along the x axis for each object")
            .def("update_y_pos_ranks",
                    &CollisionSystem::update_y_pos_ranks,
                    "Computes the rank along the y axis for each object")
            .def("update_z_pos_ranks",
                    &CollisionSystem::update_z_pos_ranks,
                    "Computes the rank along the z axis for each object")
            .def("update_mortons",
                    &CollisionSystem::update_mortons,
                    "Computes new morton numbers for each object based on their ranks. This method ensures high quality trees regardless of positions of masses in the system")
            .def("update_mortons_fast",
                    &CollisionSystem::update_mortons_fast,
                    "Computes new morton numbers for each object based on their position in 3D space relative to the bounding box given. If simulating multiple groups of masses where each group is dense but far from the other groups, don't use this")
            .def("build_tree", &CollisionSystem::build_tree, "Builds the BVH tree from the morton numbers")
            .def("update_bounding_boxes", &CollisionSystem::update_bounding_boxes, "Fills the BVH tree with the bounding boxes of each position. Call this each time before calling find_collisions. Don't need to call update_tree each time.")
            .def("find_collisions", &CollisionSystem::find_collisions, "Traverses the BVH tree to find all collisions.")
            .def("find_collisions_N2", &CollisionSystem::find_collisions_N2, "Does a simple brute force collision search (O(N^2) complexity). This is faster than tree construction for systems with only a few objects")
            .def("sync_collisions_to_host", &CollisionSystem::sync_collisions_to_host);
}
