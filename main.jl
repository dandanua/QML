using LinearAlgebra
using SparseArrays

# using Flux

⊗ = kron

# LITTLE ENDIAN !  
# 1st qubit = far right

# if IittleEndian then the last qubit is the control
function Controlled(gate::AbstractMatrix)
  sz = size(gate)
  n = sz[1]
  ret = spzeros(ComplexF64, 2*n, 2*n)
  ret[1:n,1:n] += I
  ret[n+1:2*n,n+1:2*n] = gate
  return ret
end

# returns sparse permutation matrix P of width 2^pad
# input indices of qubits are 1-based
# if G acts on qubits 1,2,... then P^(-1)*G*P acts on ind[1], ind[2],...
# LittleEndian!
function Permute(ind::Array{Int}; pad::Int=0)
  # assert maximum(ind) <= pad
  # assert ind[i] all different
  
  k = length(ind)
  pad = max(pad, maximum(ind))
  
  # reconstruct full permutation sigma
  sigma = append!(copy(ind), zeros(Int64, pad-k))
  # rest = setdiff(1:pad, ind)
  rest = filter(x -> !(x in ind), 1:pad)
  sigma[k+1:pad] = rest  
  # todo: more stable sigma?

  # println(sigma)

  rows = zeros(Int64, 2^pad)
  cols = zeros(Int64, 2^pad)
  # vals = zeros(ComplexF64, 2^pad)
  vals = zeros(Int64, 2^pad)

  function permute_binary(b::Int64)
    ret = 0

    b -= 1

    for i=1:pad
      digit = b & 1
      ret += digit << (sigma[i]-1)
      b = b >> 1
    end

    return ret+1
  end

  for b in 1:2^pad
    rows[b] = b
    cols[b] = permute_binary(b)
    vals[b] = 1
  end 

  # println("cols", cols)

  return sparse(rows, cols, vals)
end

function ApplyGateOn(gate::AbstractMatrix, ind::Array{Int}; pad::Int=0)
  # assert gate is acting on length(ind) qubits

  k = length(ind)
  pad = max(pad, maximum(ind))

  # w = size(gate)[1]
  P = Permute(ind, pad=pad)

  ret = gate
  for i in k+1:pad
    ret = [1 0; 0 1] ⊗ ret
  end

  return P'*ret*P
end

# https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.rx
function Rx(θ::Real)
  return [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
end

# https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.ry
function Ry(θ::Real)
  return [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
end

# https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.rz
function Rz(θ::Real)
  return [exp(-im*θ/2) 0; 0 exp(im*θ/2)]
end

# https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.machinelearning.controlledrotation
# little endian!
# function ControlledRotation(ind::Tuple{Int, Array{}}, Pauli::Function,  param::Real)
function ControlledRotation(target::Int, control::Int, Pauli::Function, param::Real; pad::Int=0)
  # indices are 1 based
  # id = [1 0; 0 1]

  if target==control
    return ApplyGateOn( Pauli(param), [target], pad=pad)
    # return Rz(param)
  else 
    C = Controlled(Pauli(param))
    return ApplyGateOn( C, [target, control], pad=pad)
  end
    
end


