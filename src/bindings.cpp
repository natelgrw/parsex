/*
 * bindings.cpp
 * 
 * Author: natelgrw
 * Last Edited: 02/25/2026
 * 
 * Implementation file for Python bindings.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "circuit_topology.hpp"
#include "cpu/cpu_solver.hpp"

namespace py = pybind11;

// wrapper for ParameterBatch to accept numpy arrays directly
class ParameterBatchWrapper {
public:
    ParameterBatch batch;
    
    ParameterBatchWrapper(size_t size, int num_r, int num_v, int num_i) 
        : batch(size, num_r, num_v, num_i) {}

    // expose memory buffers for zero-copy numpy integration
    void set_r_values(py::array_t<double> array) {
        py::buffer_info buf = array.request();
        if (buf.size != batch.r_values.size())
            throw std::runtime_error("R values size mismatch");
        std::memcpy(batch.r_values.data(), buf.ptr, buf.size * sizeof(double));
    }
    
    void set_v_values(py::array_t<double> array) {
        py::buffer_info buf = array.request();
        if (buf.size != batch.v_values.size())
            throw std::runtime_error("V values size mismatch");
        std::memcpy(batch.v_values.data(), buf.ptr, buf.size * sizeof(double));
    }
    
    void set_i_values(py::array_t<double> array) {
        py::buffer_info buf = array.request();
        if (buf.size != batch.i_values.size())
            throw std::runtime_error("I values size mismatch");
        std::memcpy(batch.i_values.data(), buf.ptr, buf.size * sizeof(double));
    }
};

// wrapper for solve_batch to return a nicely formatted numpy array or dict 
// actually, returning a list of dicts is slow.
// for ML, we want a flattened tensor of output voltages.
// shape: (batch_size, num_nodes - 1)
py::array_t<double> solve_batch_ml(const CircuitTopology& topo, const ParameterBatchWrapper& params) {
    size_t batch_size = params.batch.batch_size;
    int num_nodes = topo.get_num_nodes();
    int varsToReturn = num_nodes - 1; // assuming 1 ground
    
    if (topo.ground_idx == -1) varsToReturn = num_nodes;

    // allocate numpy array for results
    auto result_array = py::array_t<double>({ (int)batch_size, varsToReturn });
    py::buffer_info buf = result_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // run C++ solver
    std::vector<SimulationResult> results = CpuSolver::solve_batch(topo, params.batch);

    // pack into numpy array
    // we will extract nodes in order of topo.node_names (skipping ground)
    for (size_t b = 0; b < batch_size; ++b) {
        int v_idx = 0;
        for (int n = 0; n < num_nodes; ++n) {
            if (n == topo.ground_idx) continue;
            std::string name = topo.node_names[n];
            ptr[b * varsToReturn + v_idx] = results[b].node_voltages[name];
            v_idx++;
        }
    }

    return result_array;
}

PYBIND11_MODULE(parsex, m) {
    m.doc() = "Parsex HPC Circuit Simulator Bindings";

    py::class_<CircuitTopology>(m, "CircuitTopology")
        .def(py::init<>())
        .def("load_from_json", &CircuitTopology::load_from_json)
        .def_property_readonly("num_nodes", &CircuitTopology::get_num_nodes)
        .def_readonly("num_resistors", &CircuitTopology::num_resistors)
        .def_readonly("num_voltage_sources", &CircuitTopology::num_voltage_sources)
        .def_readonly("num_current_sources", &CircuitTopology::num_current_sources);

    py::class_<ParameterBatchWrapper>(m, "ParameterBatch")
        .def(py::init<size_t, int, int, int>())
        .def("set_r_values", &ParameterBatchWrapper::set_r_values)
        .def("set_v_values", &ParameterBatchWrapper::set_v_values)
        .def("set_i_values", &ParameterBatchWrapper::set_i_values);

    m.def("solve_batch", &solve_batch_ml, "Solve a batch of circuits and return voltages as numpy array");
}
