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

#include "bench.hpp"

#define MFEM_DEBUG_COLOR 218
#include "general/debug.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/tmop.hpp"
#include <cassert>
#include <memory>
#include <cmath>
#include <string>

struct TMOP
{
   const int n;
   const int nx, ny, nz, check, p, q, dim = 3;
   Mesh mesh;
   TMOP_Metric_302 metric;
   TargetConstructor::TargetType target_t;
   TargetConstructor target_c;
   H1_FECollection fec;
   FiniteElementSpace fes;
   const Operator *R;
   const IntegrationRule *ir;
   TMOP_Integrator nlfi;
   const int dofs;
   GridFunction x,y;
   Vector de,xe,ye;
   double mdof;

   TMOP(int p, int c, bool p_eq_q):
      n((assert(c>=p), c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check((//dbg("p:%d, c:%d, n:%p, %dx%dx%d",p,c,n,nx,ny,nz),
               assert(nx>0 && ny>0 && nz>0),1)),
      p(p), q(2*p + (p_eq_q ? 0 : 2)),
      mesh(Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON)),
      target_t(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE),
      target_c(target_t),
      fec(p, dim),
      fes(&mesh, &fec, dim),
      R(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      ir(&IntRules.Get(fes.GetFE(0)->GetGeomType(), q)),
      nlfi(&metric, &target_c),
      dofs(fes.GetVSize()),
      x(&fes),
      y(&fes),
      de(R->Height(), Device::GetMemoryType()),
      xe(R->Height(), Device::GetMemoryType()),
      ye(R->Height(), Device::GetMemoryType()),
      mdof(0.0)
   {
      /*
        dbg("p:%d, c:%d, n:%d, c_MOD_p:%d [%dx%dx%d] %d * %d * %d <?= %d * %d * %d",
            p, c, n, c%p,
            nx , ny , nz,
            p*nx , p*ny , p*nz,
            c, c ,c);*/
      MFEM_VERIFY(p*nx * p*ny * p*nz <= c*c*c, "Error");
      MFEM_VERIFY(p*(nx+1) * p*(ny+1) * p*nz > c*c*c, "Error");
      MFEM_VERIFY(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c, "Error");
      mesh.SetNodalGridFunction(&x);
      target_c.SetNodes(x);

      R->Mult(x, xe);
      ye = 0.0;

      nlfi.SetIntegrationRule(*ir);
      nlfi.AssemblePA(fes);
      nlfi.AssembleGradPA(xe,fes);

      tic_toc.Clear();
   }

   void AddMultPA()
   {
      tic_toc.Start();
      nlfi.AddMultPA(xe,ye);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void AddMultGradPA()
   {
      const bool fast = Device::FastKernelsEnabled();
      tic_toc.Start();
      if (!fast) { nlfi.AddMultGradPA(xe,ye); }
      else { nlfi.AddMultGradPA(x,y); }
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   void GetLocalStateEnergyPA()
   {
      tic_toc.Start();
      const double energy = nlfi.GetLocalStateEnergyPA(xe);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      MFEM_CONTRACT_VAR(energy);
      mdof += 1e-6 * dofs;
   }

   void AssembleGradDiagonalPA()
   {
      tic_toc.Start();
      nlfi.AssembleGradDiagonalPA(de);
      MFEM_DEVICE_SYNC;
      tic_toc.Stop();
      mdof += 1e-6 * dofs;
   }

   double Mdof() const { return mdof; }

   double Mdofs() const { return mdof / tic_toc.RealTime(); }
};

// The different orders the tests can run
#define P_ORDERS {1,2,3,4}

#define N_SIDES {4,6,8,12,16,20,24,32,48,64}

// P_EQ_Q selects the D1D & Q1D to use instantiated kernels
//  P_EQ_Q: 0x22, 0x33, 0x44, 0x55
// !P_EQ_Q: 0x23, 0x34, 0x45, 0x56
#define P_EQ_Q {false,true}

/**
 * @brief The Kernel bm::Fixture struct
 */
struct Kernel: public bm::Fixture
{
   std::unique_ptr<TMOP> ker;
   ~Kernel() { assert(ker == nullptr); }

   using bm::Fixture::SetUp;
   void SetUp(const bm::State& state) BENCHMARK_OVERRIDE
   {
      const int p = state.range(2);
      const bool p_eq_q = state.range(0);
      const int ref_factor = state.range(1);
      ker.reset(new TMOP(p, ref_factor, p_eq_q)); }

   using bm::Fixture::TearDown;
   void TearDown(const bm::State &) BENCHMARK_OVERRIDE { ker.reset(); }
};

/**
  Fixture kernels definitions and registrations
*/
#define BENCHMARK_TMOP_F(Bench)\
BENCHMARK_DEFINE_F(Kernel,Bench)(bm::State &state){\
   assert(ker.get());\
   while (state.KeepRunning()) { ker->Bench(); }\
   state.counters["MDof"] = bm::Counter(ker->Mdof(), bm::Counter::kIsRate);\
   state.counters["MDof/s"] = bm::Counter(ker->Mdofs());}\
 BENCHMARK_REGISTER_F(Kernel,Bench)->ArgsProduct({P_EQ_Q, N_SIDES, P_ORDERS})->Unit(bm::kMicrosecond);
/// creating/registering, not used
//BENCHMARK_TMOP_F(AddMultPA)
//BENCHMARK_TMOP_F(AddMultGradPA)
//BENCHMARK_TMOP_F(GetLocalStateEnergyPA)
//BENCHMARK_TMOP_F(AssembleGradDiagonalPA)

/**
  Kernels definitions and registrations
*/
#define BENCHMARK_TMOP(Bench)\
static void Bench(bm::State &state){\
   const int p = state.range(2);\
   const int c = state.range(1);\
   const bool q = state.range(0);\
   TMOP ker(p, c, q); \
   while (state.KeepRunning()) { ker.Bench(); }\
   state.SetItemsProcessed(ker.dofs*state.iterations()); \
   char label[64]; snprintf(label, 64, "%8d %.5f", ker.dofs, ker.Mdofs()); \
   state.SetLabel(label); \
   state.counters["p"] = bm::Counter(p); \
   state.counters["Dofs"] = bm::Counter(ker.dofs); \
   state.counters["MDof"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate); \
   state.counters["MDof/s"] = bm::Counter(ker.Mdofs());}\
BENCHMARK(Bench)->ArgsProduct({P_EQ_Q, N_SIDES, P_ORDERS})->Unit(bm::kMicrosecond);//->Threads(1);
/// creating/registering
BENCHMARK_TMOP(AddMultPA)
BENCHMARK_TMOP(AddMultGradPA)
BENCHMARK_TMOP(GetLocalStateEnergyPA)
BENCHMARK_TMOP(AssembleGradDiagonalPA)

/**
 * @brief main entry point
 * --benchmark_filter=AddMultPA/4
 * --benchmark_context=device=cuda
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR(bm::ConsoleReporter::OO_Tabular);
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
