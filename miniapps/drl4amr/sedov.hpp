// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#ifndef SEDOV_HPP
#define SEDOV_HPP

#include "config/config.hpp"
#include "mfem.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define PFesGetParMeshGetComm0(pfes) pfes.GetParMesh()->GetComm()
#else
typedef int MPI_Comm;
typedef int HYPRE_Int;
#define ParMesh Mesh
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Finalize()
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_INT int
#define MPI_LONG long
#define HYPRE_MPI_INT int
#define MPI_DOUBLE double
template<typename T>
void MPI_Reduce_(T *src, T *dst, const int n)
{
   for (int i=0; i<n; i++) { dst[i] = src[i]; }
}
#define MPI_Reduce(src, dst, n, T,...) MPI_Reduce_<T>(src,dst,n)
class MPI_Session
{
public:
   MPI_Session() {}
   MPI_Session(int argc, char **argv) {}
   bool Root() { return true; }
   int WorldRank() { return 0; }
   int WorldSize() { return 1; }
};
#endif

namespace mfem
{

namespace hydrodynamics
{
class LagrangianHydroOperator;
}

class sedov
{
private:
   MPI_Session *mpi;
   Device device;
   int myid;
   const char *mesh_file = "data/cube01_hex.mesh";
   int rs_levels = 0;
   const int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-14;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *dev_opt = "cpu";
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = -1;
   double blast_energy = 0.25;
   Mesh *mesh;
   int dim;
   ParMesh *pmesh;
   L2_FECollection *L2FEC;
   H1_FECollection *H1FEC;
   ParFiniteElementSpace *L2FESpace, *H1FESpace;
   Array<int> ess_tdofs;
   HYPRE_Int H1GTVSize ;
   HYPRE_Int L2GTVSize ;
   int H1Vsize;
   int L2Vsize ;
   BlockVector *S;
   ParGridFunction x_gf, v_gf, e_gf;
   ParGridFunction *rho;
   mfem::hydrodynamics::LagrangianHydroOperator *oper;
   double t, dt, t_old;
   int ti;
   BlockVector *S_old;
   ODESolver *ode_solver;
   bool last_step = false;
   int steps = 0;
public:
   sedov(int argc, char *argv[]);
   ~sedov();
   void Step();
};

} // namespace mfem
#endif // SEDOV_HPP
