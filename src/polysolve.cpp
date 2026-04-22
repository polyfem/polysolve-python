#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

#include <polysolve/Types.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace
{
    using Problem = polysolve::nonlinear::Problem;
    using TVector = Problem::TVector;
    using TMatrix = Problem::TMatrix;
    using THessian = Problem::THessian;

    std::shared_ptr<spdlog::logger> python_logger(const spdlog::level::level_enum log_level)
    {
        auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cout, true);
        auto logger = std::make_shared<spdlog::logger>("polysolve", sink);
        logger->set_level(log_level);
        logger->set_pattern("%v");
        return logger;
    }

    py::function optional_override(const Problem *self, const char *name)
    {
        return py::get_override(self, name);
    }

    py::function required_override(const Problem *self, const char *name)
    {
        py::function f = optional_override(self, name);
        if (!f)
            throw std::runtime_error(std::string("Python Problem subclass must implement '") + name + "'");
        return f;
    }

    void cast_gradient(const py::object &obj, const Eigen::Index expected_size, TVector &grad)
    {
        grad = obj.cast<TVector>();
        if (grad.size() != expected_size)
        {
            throw std::runtime_error(
                "gradient returned vector with size " + std::to_string(grad.size())
                + ", expected " + std::to_string(expected_size));
        }
    }

    void cast_sparse_hessian(const py::object &obj, const Eigen::Index expected_size, THessian &hessian)
    {
        try
        {
            hessian = obj.cast<THessian>();
        }
        catch (const py::cast_error &)
        {
            hessian = obj.cast<TMatrix>().sparseView();
        }

        if (hessian.rows() != expected_size || hessian.cols() != expected_size)
        {
            throw std::runtime_error(
                "hessian returned matrix with shape (" + std::to_string(hessian.rows()) + ", "
                + std::to_string(hessian.cols()) + "), expected (" + std::to_string(expected_size)
                + ", " + std::to_string(expected_size) + ")");
        }
    }

    void cast_dense_hessian(const py::object &obj, const Eigen::Index expected_size, TMatrix &hessian)
    {
        try
        {
            hessian = obj.cast<TMatrix>();
        }
        catch (const py::cast_error &)
        {
            hessian = TMatrix(obj.cast<THessian>());
        }

        if (hessian.rows() != expected_size || hessian.cols() != expected_size)
        {
            throw std::runtime_error(
                "hessian returned matrix with shape (" + std::to_string(hessian.rows()) + ", "
                + std::to_string(hessian.cols()) + "), expected (" + std::to_string(expected_size)
                + ", " + std::to_string(expected_size) + ")");
        }
    }

    class PyProblem : public Problem
    {
    public:
        using Problem::Problem;

        void init(const TVector &x0) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "init"))
                f(x0);
        }

        Scalar value(const TVector &x) override
        {
            py::gil_scoped_acquire gil;
            return required_override(this, "value")(x).cast<Scalar>();
        }

        void gradient(const TVector &x, TVector &grad) override
        {
            py::gil_scoped_acquire gil;
            cast_gradient(required_override(this, "gradient")(x), x.size(), grad);
        }

        void hessian(const TVector &x, TMatrix &hessian) override
        {
            py::gil_scoped_acquire gil;
            cast_dense_hessian(required_override(this, "hessian")(x), x.size(), hessian);
        }

        void hessian(const TVector &x, THessian &hessian) override
        {
            py::gil_scoped_acquire gil;
            cast_sparse_hessian(required_override(this, "hessian")(x), x.size(), hessian);
        }

        bool is_step_valid(const TVector &x0, const TVector &x1) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "is_step_valid"))
                return f(x0, x1).cast<bool>();
            return Problem::is_step_valid(x0, x1);
        }

        double max_step_size(const TVector &x0, const TVector &x1) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "max_step_size"))
                return f(x0, x1).cast<double>();
            return Problem::max_step_size(x0, x1);
        }

        void line_search_begin(const TVector &x0, const TVector &x1) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "line_search_begin"))
                f(x0, x1);
        }

        void line_search_end() override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "line_search_end"))
                f();
        }

        void set_project_to_psd(bool val) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "set_project_to_psd"))
                f(val);
        }

        void solution_changed(const TVector &new_x) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "solution_changed"))
                f(new_x);
        }

        bool after_line_search_custom_operation(const TVector &x0, const TVector &x1) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "after_line_search_custom_operation"))
                return f(x0, x1).cast<bool>();
            return Problem::after_line_search_custom_operation(x0, x1);
        }

        bool stop(const TVector &x) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "stop"))
                return f(x).cast<bool>();
            return Problem::stop(x);
        }

        void post_step(const polysolve::nonlinear::PostStepData &data) override
        {
            py::gil_scoped_acquire gil;
            if (py::function f = optional_override(this, "post_step"))
                f(data.iter_num, data.solver_info, data.x, data.grad);
        }
    };

} // namespace

PYBIND11_MODULE(polysolve, m)
{
    using namespace polysolve;
    namespace nonlinear = polysolve::nonlinear;
    namespace nl = nlohmann;

    m.doc() = "Python bindings for PolySolve nonlinear optimization.";

    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("trace", spdlog::level::trace)
        .value("debug", spdlog::level::debug)
        .value("info", spdlog::level::info)
        .value("warn", spdlog::level::warn)
        .value("error", spdlog::level::err)
        .value("critical", spdlog::level::critical)
        .value("off", spdlog::level::off);

    py::class_<Problem, PyProblem>(m, "Problem")
        .def(py::init<>())
        .def("value", [](Problem &problem, const TVector &x) {
            return problem.value(x);
        })
        .def("gradient", [](Problem &problem, const TVector &x) {
            TVector grad;
            problem.gradient(x, grad);
            return grad;
        })
        .def("hessian", [](Problem &problem, const TVector &x) {
            THessian hessian;
            problem.hessian(x, hessian);
            return hessian;
        });

    m.def(
        "minimize",
        [](Problem &problem,
           const TVector &x0,
           const py::dict &solver_params,
           const py::dict &linear_solver_params,
           const double characteristic_length,
           const spdlog::level::level_enum log_level,
           const bool strict_validation) {
            py::scoped_ostream_redirect stdout_redirect(
                std::cout, py::module_::import("sys").attr("stdout"));
            auto logger = python_logger(log_level);

            nl::json jsolver_params = solver_params.cast<nl::json>();
            nl::json jlinear_solver_params = linear_solver_params.cast<nl::json>();

            TVector x = x0;

            auto solver = nonlinear::Solver::create(
                jsolver_params,
                jlinear_solver_params,
                characteristic_length,
                *logger,
                strict_validation);
            solver->minimize(problem, x);

            return std::make_pair(x, solver->info());
        },
        "Minimize a nonlinear optimization problem.",
        py::arg("problem"),
        py::arg("x0"),
        py::arg("solver_params") = py::dict(),
        py::arg("linear_solver_params") = py::dict(),
        py::arg("characteristic_length") = 1.0,
        py::arg("log_level") = spdlog::level::info,
        py::arg("strict_validation") = true);
}
