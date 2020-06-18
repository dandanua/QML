using Optim
# using Zygote

const eps = 1e-7

σ(x) = 1 / (1 + exp(-x))
mse(ŷ, y) = sum((ŷ .- y).^2) / length(y)

function misses(predicted::AbstractVecOrMat, data::AbstractVecOrMat)::Float64
  err = 0.0
  n = length(data)
  for i in 1:n
    err += abs2(predicted[i] - data[i]) > eps ? 1 : 0
  end
  
  return err / n
end

include("main.jl")
include("enc.jl")
include("pre.jl")  

function optim()
  
  
  function model(input::Array{Float64,1}, param::Array{Float64,1})::Float64
    ϕ = param[1:end-1]
    b = param[end]

    # 1.0, 0.2471
    # 3.1121371444218893, 0.001393518160188087
    
    M = I
    M *= ControlledRotation(1, 1, Ry, ϕ[1], pad=2)
    M *= ControlledRotation(2, 2, Ry, ϕ[2])
    M *= ControlledRotation(1, 2, Ry, ϕ[3])
    M *= ControlledRotation(2, 1, Ry, ϕ[4])

    r = M * input

    # return abs2(r[2]) + b > 0.5 ? 1 : 0
    
    ret = abs2(r[2]) + b
    # ret = σ( (ret - 0.5)*1000 )       
    return ret
  end
  
  tr = encode(pre())
  tr_x = [d[1] for d in tr]
  tr_y = [d[2] for d in tr]


  function target(param::Array{Float64,1})::Float64
    prob_y = map( λ->model(λ, param), tr_x) 
    return mse(prob_y, tr_y)
  end

  function target2(param::Array{Float64,1})::Float64
    prob_y = map( λ-> σ( (model(λ, param) - 0.5)*1000 ) , tr_x) 
    return mse(prob_y, tr_y)
  end
  
  # p_init = [3.0, 3.0, 3.0, 3.0, 0.0]
  # p_init = [0.0, 0.0, 0.0, 0.0, 0.0]
  p_init = [1.2524733645306474, -1.141071755919144, -0.08147920673002419, -1.2266097709687462, 0.2010126029766885]
  # p_init = [3.323096449858975, 0.045860433297251896, 2.248264115129544, 2.5039894584527858, 0.12339406935001004]

  @show target2(p_init)
  # prob_y = map( λ->model(λ, p_init), tr_x) 
  prob_y = map( λ-> σ( (model(λ, p_init) - 0.5)*1000 ) , tr_x) 
  predicted = map(x -> x>0.5 ? 1 : 0, prob_y)
  @show misses(predicted, tr_y)

  # grad = target'
  # grad!(stor,x) = copyto!(stor,grad(x))

  manif = Optim.Flat()
  optimizer = Optim.ConjugateGradient(manifold=manif)
  # optimizer = Optim.GradientDescent(manifold=manif)
  # optimizer = Optim.LBFGS(manifold=manif)
  # res = optimize(target, p_init, optimizer; autodiff = :forward)
  # res = optimize(target, grad!, p_init, optimizer)
  res = optimize(target2, p_init, optimizer, Optim.Options(g_tol = 1e-5, allow_f_increases = false, show_trace=false))

  bestmn = Optim.minimum(res)
  bestmnz = Optim.minimizer(res)

  @show res
  @show bestmn
  @show bestmnz
  
  @show target2(bestmnz)
  prob_y = map( λ-> σ( (model(λ, bestmnz) - 0.5)*1000 ) , tr_x) 
  predicted = map(x -> x>0.5 ? 1 : 0, prob_y)
  @show misses(predicted, tr_y)

end

@time optim()