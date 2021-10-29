// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../tmop_tools.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

template <typename T = double>
struct SMem
{
   const int bid, grid, msz;
   T *gmem;
   SMem(T *gmem, int grid, int msz):
      bid(MFEM_BLOCK_ID(x)),
      grid(grid),
      msz(msz),
      gmem(gmem + msz*(grid+bid)) { }

   MFEM_DEVICE inline operator T *() noexcept { return (T*) gmem; }

   template<typename U> static MFEM_DEVICE
   U *alloc(U* &gmem, size_t size) noexcept
   {
      U* base = gmem;
      return (gmem += size, base);
   }
};

//MFEM_REGISTER_TMOP_KERNELS(double, MinDetJpr_Kernel_3D,
template<int T_D1D = 0, int T_Q1D = 0, bool SMEM = true>
double MinDetJpr_Kernel_3D(const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const Vector &x_,
                           Vector &DetJ,
                           const int d1d = 0,
                           const int q1d = 0,
                           Vector *d_buff = nullptr)
{
   constexpr int DIM = 3;
   static constexpr int MQ1 = T_Q1D ? T_Q1D : 1;
   static constexpr int MD1 = T_D1D ? T_D1D : 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const int MSZ = + 3*D1D*D1D*D1D
                   + 6*D1D*D1D*Q1D
                   + 9*D1D*Q1D*Q1D
                   + 9*Q1D*Q1D*Q1D;
   static constexpr int GRID = SMEM ? 0 : 128;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, DIM, NE);

   auto E = Reshape(DetJ.Write(), Q1D, Q1D, Q1D, NE);

   double *GMEM = nullptr;
   if (!SMEM)
   {
      d_buff->SetSize(MSZ*GRID);
      GMEM = d_buff->Write();
   }

   //MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   MFEM_FORALL_3D_GRID(e, NE, Q1D, Q1D, Q1D, GRID,
   {

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double SM0[SMEM?3*D1D*D1D*D1D:1];
      MFEM_SHARED double SM1[SMEM?6*D1D*D1D*Q1D:1];
      MFEM_SHARED double SM2[SMEM?9*D1D*Q1D*Q1D:1];
      MFEM_SHARED double SM3[SMEM?9*Q1D*Q1D*Q1D:1];

      double *gmm = SMem<>(GMEM, GRID, MSZ);
      double *lm0 = SMEM ? SM0 : SMem<>::alloc(gmm, 3*D1D*D1D*D1D);
      double *lm1 = SMEM ? SM1 : SMem<>::alloc(gmm, 6*D1D*D1D*Q1D);
      double *lm2 = SMEM ? SM2 : SMem<>::alloc(gmm, 9*D1D*Q1D*Q1D);
      double *lm3 = SMEM ? SM3 : SMem<>::alloc(gmm, 9*Q1D*Q1D*Q1D);

      double (*DDD)[MD1*MD1*MD1] = (double (*)[MD1*MD1*MD1]) (lm0);
      double (*DDQ)[MD1*MD1*MQ1] = (double (*)[MD1*MD1*MQ1]) (lm1);
      double (*DQQ)[MD1*MQ1*MQ1] = (double (*)[MD1*MQ1*MQ1]) (lm2);
      double (*QQQ)[MQ1*MQ1*MQ1] = (double (*)[MQ1*MQ1*MQ1]) (lm3);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      //constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      //constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      //MFEM_SHARED double DDD[3][MD1*MD1*MD1];
      //MFEM_SHARED double DDQ[6][MD1*MD1*MQ1];
      //MFEM_SHARED double DQQ[9][MD1*MQ1*MQ1];
      //MFEM_SHARED double QQQ[9][MQ1*MQ1*MQ1];

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double J[9];
               kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz,QQQ,J);
               E(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });

   return DetJ.Min();
}

double TMOPNewtonSolver::MinDetJpr_3D(const FiniteElementSpace *fes,
                                      const Vector &X) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R = fes->GetElementRestriction(ordering);
   static Vector XE(R->Height(), Device::GetDeviceMemoryType());
   XE.UseDevice(true);
   R->Mult(X, XE);

   const DofToQuad &maps = fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int NE = fes->GetMesh()->GetNE();
   const int NQ = ir.GetNPoints();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<double> &B = maps.B;
   const Array<double> &G = maps.G;

   static Vector E(NE*NQ);
   E.UseDevice(true);

   static Vector d_buffer; d_buffer.UseDevice(true);

   switch (id)
   {
      case 0x24: return MinDetJpr_Kernel_3D<2,4>(NE,B,G,XE,E);
      case 0x33: return MinDetJpr_Kernel_3D<3,3>(NE,B,G,XE,E);
      case 0x35: return MinDetJpr_Kernel_3D<3,5>(NE,B,G,XE,E);
      case 0x36: return MinDetJpr_Kernel_3D<3,6>(NE,B,G,XE,E);
      default:
         return MinDetJpr_Kernel_3D<0,0,false>(NE,B,G,XE,E,D1D,Q1D,&d_buffer);
   }
   //MFEM_LAUNCH_TMOP_KERNEL(MinDetJpr_Kernel_3D,id,NE,B,G,XE,E);
   assert(false);
}

} // namespace mfem
