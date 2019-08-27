using Plots
using NLsolve: nlsolve, converged
using OrdinaryDiffEq: ODEProblem, Rodas4, init, solve!,solve, step!, reinit!,savevalues!, u_modified!, ODEFunction

struct RootRhs
    rhs
end

function (rr::RootRhs)(x)
    dx = similar(x)
    rr.rhs(dx, x, nothing, 0.)
    dx
end

struct NetworkEq
    P
end

NetworkEq(; P) = NetworkEq(P)

function (neq::NetworkEq)(dx, x, params ,t)
    admittance = -1im/0.02
    Y_matrix = [-admittance admittance;
                admittance -admittance]
    H_1 =5
    P_2 = -1
    H_2 =5
    D_1=0.1
    D_2=1
    Ω_H_1 = 2*π*50/H_1
    Ω_H_2 = 2*π*50/H_2
    Γ=0.1
    V=1.
    u_1 = x[1] + x[2] * im
    ω_1 = x[3]
    u_2 = x[4] + x[5] * im
    ω_2 = x[6]
    u = [u_1;u_2]
    i = -Y_matrix*u
    i_1 = i[1]
    i_2 = i[2]
    p = real(u .* conj(i))
    dϕ_1 = ω_1
    if (params != nothing)
        dω_1 = (((neq.P*params) - D_1 * ω_1) - p[1]) * Ω_H_1
    else
        dω_1 = ((neq.P - D_1 * ω_1) - p[1]) * Ω_H_1
    end
    v_1 = abs(u_1)
    dv_1 = -Γ * (v_1 - V)
    du_1 = (u_1 / v_1) * dv_1 + u_1 * im * dϕ_1

    dϕ_2 = ω_2
    dω_2 = ((P_2 - D_2 * ω_2) - p[2]) * Ω_H_2
    v_2 = abs(u_2)
    dv_2 = -Γ * (v_2 - V)
    du_2 = (u_2 / v_2) * dv_2 + u_2 * im * dϕ_2

    dx[1] = real(du_1)
    dx[2] = imag(du_1)
    dx[3] = dω_1
    dx[4] = real(du_2)
    dx[5] = imag(du_2)
    dx[6] = dω_2
    return nothing
end

## this works
function simulate_with_param(rhs, x0, timespan,tspan_fault)

    problem = ODEProblem{true}(rhs, x0, timespan, 1.0)
    integrator = init(problem, Rodas4(autodiff=false))

    step!(integrator, tspan_fault[1], true)

    ## update integrator with error
    integrator.p = 0.9

    step!(integrator, tspan_fault[2], true)

    ## update integrator, clear error
    integrator.p = 1.0

    solve!(integrator)

    return integrator.sol
end

## this works, but the code looks hacky. Can't we do any better?
function simulate_hack(rhs1,rhs2, x0, timespan,tspan_fault)
    problem2 = ODEProblem{true}(rhs2,x0,(first(timespan),tspan_fault[2]))
    fault_integrator = init(problem2, Rodas4(autodiff=false))

    reinit!(fault_integrator, x0, t0=tspan_fault[1], tf=tspan_fault[2], erase_sol=false)
    savevalues!(fault_integrator)
    solve!(fault_integrator)

    problem1 = ODEProblem{true}(rhs1, fault_integrator.u, (tspan_fault[2], last(timespan)))
    integrator = init(problem1, Rodas4(autodiff=false))

    integrator.sol = fault_integrator.sol
    integrator.saveiter = fault_integrator.saveiter
    integrator.saveiter_dense = fault_integrator.saveiter_dense
    integrator.success_iter = fault_integrator.success_iter

    solve!(integrator)

    return integrator.sol
end

## this also works but is very sloooooow
function simulate_switch_rhs(rhs1, rhs2, x0, timespan,tspan_fault)

    ode_f1 = ODEFunction(rhs1)
    problem = ODEProblem{true}(ode_f1, x0, timespan)
    integrator = init(problem, Rodas4(autodiff=false))

    step!(integrator, tspan_fault[1], true)

    ## update integrator with error
    integrator.f = ODEFunction(rhs2)
    u_modified!(integrator, true)

    step!(integrator, tspan_fault[2], true)

    ## update integrator, clear error
    integrator.f = ode_f1

    solve!(integrator)

    return integrator.sol
end


function find_operationpoint(rhs)
    ic_guess = ones(6) ##hardcoded, 2 nodes, each with 3 vars
    root_rhs=RootRhs(rhs)
    nl_res = nlsolve(root_rhs, ic_guess)
    if converged(nl_res) == true
        return nl_res.zero
    else
        throw("Failed to find initial conditions on the constraint manifold!")
    end
end


function run_sim_params()
    rhs_stable = NetworkEq(P=1)
    operationpoint = find_operationpoint(rhs_stable)
    tspan_sim =   (0.,1.)
    tspan_fault = (0.1,1)

    sol = simulate_with_param(rhs_stable,
        operationpoint,
        tspan_sim,
        tspan_fault)
end

function run_sim_hack()
    rhs_stable = NetworkEq(P=1)
    rhs_fault = NetworkEq(P=0.9)
    operationpoint = find_operationpoint(rhs_stable)
    tspan_sim =   (0.,1.)
    tspan_fault = (0.1,1)

    sol = simulate_hack(rhs_stable,rhs_fault,
        operationpoint,
        tspan_sim,
        tspan_fault)
end

function run_sim_switch_rhs()
    rhs_stable = NetworkEq(P=1)
    rhs_fault = NetworkEq(P=0.9)
    operationpoint = find_operationpoint(rhs_stable)
    tspan_sim =   (0.,1.)
    tspan_fault = (0.1,1)

    sol = simulate_switch_rhs(rhs_stable,rhs_fault,
        operationpoint,
        tspan_sim,
        tspan_fault)
end

#-
sol1 = run_sim_hack()
plot(sol1, vars=[3,6])
#-
sol2 = run_sim_params()
plot(sol2, vars=[3,6])
#-
sol3 = run_sim_switch_rhs()
plot(sol3, vars=[3,6])

#-
using BenchmarkTools
@benchmark run_sim_hack()
#-
@benchmark run_sim_params()
#-
@benchmark run_sim_switch_rhs()
