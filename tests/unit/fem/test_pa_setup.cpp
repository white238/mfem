// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "unit_tests.hpp"

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

namespace pa_setup
{

void ho_pa_diffusion_3d_checks(Mesh &mesh, ConstantCoefficient &Q,
                               const int p,
                               const int max_check_p = 3)
{
   constexpr int seed = 0x100001b3;
   const int dim = mesh.Dimension();

   H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace vfes(&mesh, &fec, dim);

   mesh.SetNodalFESpace(&vfes);
   GridFunction x(&vfes);
   mesh.SetNodalGridFunction(&x);

   Vector h0(vfes.GetNDofs());
   h0 = infinity();
   {
      Array<int> dofs;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         vfes.GetElementDofs(i, dofs);
         const double hi = mesh.GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = std::min(h0(dofs[j]), hi);
         }
      }
   }

   GridFunction rdm(&vfes);
   rdm.Randomize(seed);
   rdm -= 0.5;
   rdm /= M_PI;
   rdm.HostReadWrite();

   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < vfes.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(vfes.DofToVDof(i,d)) *= h0(i);
      }
   }

   Array<int> vdofs;
   for (int i = 0; i < vfes.GetNBE(); i++)
   {
      vfes.GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;

   const IntegrationRule &ir = IntRules.Get(Geometry::CUBE, p);

   if (p <= max_check_p) // Sanity checks PA vs FA
   {
      // Vector
      BilinearForm vpa(&vfes), vfa(&vfes);
      vpa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      vpa.AddDomainIntegrator(new VectorDiffusionIntegrator(Q));
      vfa.AddDomainIntegrator(new VectorDiffusionIntegrator(Q));
      (vpa.Assemble(), vfa.Assemble(), vfa.Finalize());
      Vector vpad(vfes.GetVSize()), vfad(vfes.GetVSize());
      (vpa.AssembleDiagonal(vpad), vfa.SpMat().GetDiag(vfad));
      const double vdpa = vpad*vpad, vdfa = vfad*vfad;
      std::cout << "[v-check] #" << p << " " << vdpa << std::endl;
      REQUIRE(MFEM_Approx(vdpa) == vdfa);

      // Scalar
      BilinearForm pa(&fes), fa(&fes);
      pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      pa.AddDomainIntegrator(new DiffusionIntegrator(Q));
      fa.AddDomainIntegrator(new DiffusionIntegrator(Q));
      (pa.Assemble(), fa.Assemble(), fa.Finalize());
      Vector pad(fes.GetVSize()), fad(fes.GetVSize());
      (pa.AssembleDiagonal(pad), fa.SpMat().GetDiag(fad));
      const double dpa = pad*pad, dfa = fad*fad;
      std::cout << "[s-check] #" << p << " " << dpa << std::endl;
      REQUIRE(MFEM_Approx(dpa) == dfa);
      return;
   }

   BilinearForm vpa(&vfes);
   vpa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   vpa.AddDomainIntegrator(new VectorDiffusionIntegrator(Q));
   vpa.Assemble();
   Vector vpad(vfes.GetVSize());
   vpa.AssembleDiagonal(vpad);
   const double vdpa = vpad*vpad;
   std::cout << "[v-kerchk] #" << p << " " << vdpa << std::endl;
   REQUIRE(MFEM_Approx(vdpa) == vdpa);

   BilinearForm pa(&fes);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   DiffusionIntegrator *di = new DiffusionIntegrator(Q);
   di->SetIntegrationRule(ir);
   pa.AddDomainIntegrator(di);
   pa.Assemble();
   Vector pad(fes.GetVSize());
   pa.AssembleDiagonal(pad);
   const double dpa = pad*pad;
   std::cout << "[s-kerchk] #" << p << " " << dpa << std::endl;
   REQUIRE(MFEM_Approx(dpa) == dpa);
}

void ho_pa_diffusion_3d_kernel(Mesh &mesh, ConstantCoefficient &Q, const int p)
{
   H1_FECollection fec(p, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   DiffusionIntegrator diffusion(Q);
   diffusion.AssemblePA(fes);
}

TEST_CASE("pad3d-checks", "[PartialAssembly] [PAD3D] [CUDA]")
{
   const double M_PI2_6 = M_PI*M_PI/6.0;
#ifdef MFEM_DEBUG
   const auto p = GENERATE(range(4, 0, -1));
#else
   const auto p = GENERATE(range(12, 0, -1));
#endif

   const int ne = 3;
   Mesh mesh(ne, ne, ne, Element::HEXAHEDRON);

   ConstantCoefficient Q(M_PI2_6);

   SECTION("3D") { ho_pa_diffusion_3d_checks(mesh, Q, p); }
   SECTION("3D") { ho_pa_diffusion_3d_kernel(mesh, Q, p); }

} // test case

} // namespace pa_setup
