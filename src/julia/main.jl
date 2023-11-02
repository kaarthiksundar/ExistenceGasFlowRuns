using GasSteadySim 
using LinearAlgebra 
using Random
using Distributions
using NLSolversBase

""" helper function to create pressure formulation initial guess from potential formulation """
function _create_initial_guess(ss_pressure::SteadySimulator, 
    ss_potential::SteadySimulator)::Vector{Float64}

    ndofs = length(ref(ss_pressure, :dof))
    guess = zeros(Float64, ndofs)

    for i in 1:length(ref(ss_potential, :dof))
        comp, id = ss_potential.ref[:dof][i]
        if comp == :node
            val = ref(ss_pressure, :is_pressure_node, id) ? ss_potential.ref[comp][id]["pressure"] : ss_potential.ref[comp][id]["potential"]
            guess[ss_pressure.ref[comp][id]["dof"]] = val
        elseif comp in [:pipe, :compressor, :valve]
            guess[ss_pressure.ref[comp][id]["dof"]] = ss_potential.ref[comp][id]["flow"]
        end 
    end 
    return guess
end 

function perturb_compressor_ratios!(data::Dict{String,Any}, lb::Float64, ub::Float64)
    c_ratios = Dict()
    for (id, compressor) in get(data, "boundary_compressor", [])
        (compressor["value"] == 1.0) && (c_ratios[id] = 1.0; continue)
        if (lb == ub)
            c_ratios[id] = lb 
        else 
            c_ratios[id] = rand(Uniform(lb, ub))
        end 
    end 
    return c_ratios
end 

function modify_data!(data::Dict{String,Any}, c_ratios::Dict)
    for (id, compressor) in get(data, "boundary_compressor", [])
        compressor["value"] = c_ratios[id]
    end 
end 

function run(; num_samples = 1000)
    folder = "../../data/GasLib-11/"
    eos = :simple_cnga 
    Random.seed!(2022) 
    
    for _ in 1:num_samples
        data = GasSteadySim._parse_data(folder)
        c_ratios = perturb_compressor_ratios!(data, 1.1, 1.4)
        modify_data!(data, c_ratios) 

        # create and solve the potential formulation
        ss_potential = initialize_simulator(data, 
            eos = eos, 
            use_potential_formulation = true, 
            potential_ratio_coefficients = [0.0, 0.0, 0.9, 0.1]) 

        df_potential = prepare_for_solve!(ss_potential)
        solver_return_potential = run_simulator!(ss_potential, df_potential, show_trace_flag=false)
        num_potential_iterations = solver_return_potential.iterations  

        # create and solve pressure formulation 
        data = GasSteadySim._parse_data(folder)
        modify_data!(data, c_ratios) 
        ss_pressure = initialize_simulator(data, eos = eos)
        df_pressure = prepare_for_solve!(ss_pressure)
        solver_return_pressure = run_simulator!(ss_pressure, df_pressure, show_trace_flag=false)
        num_pressure_iterations = solver_return_pressure.iterations  

        # create guess from potential solution for pressure formulation and solve
        x_guess = _create_initial_guess(ss_pressure, ss_potential)
        residual_of_guess = value!(df_pressure, x_guess)
        R_inf = norm(residual_of_guess, Inf)
        solver_return_pressure_with_guess = run_simulator!(ss_pressure, 
            df_pressure, x_guess = x_guess, show_trace_flag=false) 
        num_pressure_with_guess_iterations = solver_return_pressure_with_guess.iterations
        x = solver_return_pressure_with_guess.solution 

        relative_error = norm(x - x_guess, Inf) / norm(x)
        err = solver_return_pressure.solution - solver_return_pressure_with_guess.solution
        @show R_inf, relative_error, norm(err)
    end 

end 
