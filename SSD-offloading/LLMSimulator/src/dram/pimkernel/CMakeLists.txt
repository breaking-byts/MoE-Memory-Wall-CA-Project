#add_library(pimkernel OBJECT)

target_sources(
  dram PRIVATE

  pim_kernel.cpp
  Read.cpp
  Write.cpp
  GEMV.cpp
  # pimkernel/Move.cpp
  # pimkernel/Mult.cpp
  # pimkernel/Add.cpp
  # pimkernel/MAD.cpp
  # pimkernel/PMult.cpp
  # pimkernel/CMult.cpp
  # pimkernel/CAdd.cpp
  # pimkernel/CMAD.cpp
  # pimkernel/Tensor.cpp
  # pimkernel/Tensor_Square.cpp
  # pimkernel/ModUp_Evkmult.cpp
  # pimkernel/ModDownEpilogue.cpp
  # pimkernel/PMult_Accum.cpp
  # pimkernel/CMult_Accum.cpp
)



