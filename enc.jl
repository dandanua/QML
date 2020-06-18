using LinearAlgebra

⊗ = kron

function encode(s::Tuple{Array{Float64,1},Int64})
  n = sqrt(s[1][1]^2 + s[1][2]^2)
  # return (s[1]/n, s[2])
  return (s[1]/n ⊗ s[1]/n, s[2])
end

function encode(ar::Array{Tuple{Array{Float64,1},Int64}})
  return encode.(ar)  
end

