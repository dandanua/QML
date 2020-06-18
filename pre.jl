using JSON

function pre()
  # data = JSON.parsefile("training_data1.json")
  data = JSON.parsefile("training_data2.json")

  n = length(data["Labels"])

  labelled_data = Array{Tuple{Array{Float64,1},Int64}}(undef, n)

  for i in 1:n
    labelled_data[i] = ( data["Features"][i], data["Labels"][i] )   
  end

  return labelled_data
end 

# @time pre()