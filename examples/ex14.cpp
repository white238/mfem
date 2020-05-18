//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class Multigrid : public Solver
{
public:
   Multigrid() : Solver() {}

   virtual void Mult(const Vector &x_, Vector &y_) const
   {
      std::vector<const HypreParMatrix *> A;
      A.push_back(static_cast<const HypreParMatrix *>(op));

      std::vector<Vector> x;
      x.push_back(x_);

      std::vector<Vector> b;
      b.push_back(x_);

      std::vector<Vector> r;
      r.push_back(y_);

      int nlevels = P.size();

      if (P.size() == 0)
      {
         Operator *Arow = new SuperLURowLocMatrix(*A[0]);

         SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
         superlu->SetPrintStatistics(false);
         superlu->SetSymmetricPattern(true);
         superlu->SetColumnPermutation(superlu::PARMETIS);
         superlu->SetOperator(*Arow);

         superlu->Mult(x_, y_);

         delete superlu;
         delete Arow;

         // BlockILU0 S(*const_cast<HypreParMatrix *>(A[0]),
         //             cpfes->GetFE(0)->GetDof());

         // S.Mult(x_, y_);
         return;
      }

      // Build hierarchy
      for (int i = 0; i < nlevels; ++i)
      {
         A.push_back(RAP(A[i], P[i]));

         Vector w(P[i]->Width());

         P[i]->MultTranspose(x[i], w);
         x.push_back(w);

         P[i]->MultTranspose(b[i], w);
         b.push_back(w);

         P[i]->MultTranspose(r[i], w);
         r.push_back(w);
      }

      // Pre-smoothing
      for (int i = 0; i < P.size(); ++i)
      {
         // BlockILU0 S(*const_cast<HypreParMatrix *>(A[i]),
         //             cpfes->GetFE(0)->GetDof());

         HypreSmoother S(*const_cast<HypreParMatrix *>(A[i]));

         // Smooth
         S.Mult(b[i], x[i]);

         // r = b - A * x
         A[i]->Mult(x[i], r[i]);
         add(b[i], -1.0, r[i], r[i]);

         // Restrict to next coarse grid
         P[i]->MultTranspose(r[i], b[i + 1]);
      }

      {
         // Coarse solve
         Operator *Arow = new SuperLURowLocMatrix(*A[nlevels - 1]);

         SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
         superlu->SetPrintStatistics(false);
         superlu->SetSymmetricPattern(true);
         superlu->SetColumnPermutation(superlu::PARMETIS);
         superlu->SetOperator(*Arow);

         superlu->Mult(b[nlevels - 1], x[nlevels - 1]);

         delete superlu;
         delete Arow;

         // BlockILU0 S(*const_cast<HypreParMatrix *>(A[nlevels - 1]),
         //             cpfes->GetFE(0)->GetDof());

         // GMRESSolver gmres_coarse(MPI_COMM_WORLD);
         // gmres_coarse.iterative_mode = false;
         // gmres_coarse.SetPreconditioner(S);
         // gmres_coarse.SetOperator(*A[nlevels - 1]);
         // gmres_coarse.SetMaxIter(500);
         // gmres_coarse.SetRelTol(1e-12);
         // gmres_coarse.SetPrintLevel(0);
         // gmres_coarse.Mult(b[nlevels - 1], x[nlevels - 1]);
      }

      // Post-smoothing
      for (int i = nlevels - 1; i > 0; --i)
      {
         // Prolongate to next fine grid
         P[i - 1]->Mult(x[i], x[i - 1]);

         // r = b - A * x
         A[i - 1]->Mult(x[i - 1], r[i - 1]);
         add(b[i - 1], -1.0, r[i - 1], r[i - 1]);

         // BlockILU0 S(*const_cast<HypreParMatrix *>(A[i - 1]),
         //             cpfes->GetFE(0)->GetDof());

         HypreSmoother S(*const_cast<HypreParMatrix *>(A[i - 1]));

         // Smooth
         S.Mult(r[i - 1], b[i - 1]);
         x[i - 1].Add(0.9, b[i - 1]);
      }

      y_ = x[0];

      for (int i = 1; i < A.size(); ++i)
      {
         delete A[i];
      }
   }

   virtual void SetOperator(const Operator &op_)
   {
      op = &op_;
      height = op->Height();
      width = op->Width();
   }

   void AddLevel(const ParFiniteElementSpace &pfes)
   {
      if (cpfes != nullptr)
      {
         OperatorHandle Tr(Operator::Hypre_ParCSR);
         pfes.GetTrueTransferOperator(*cpfes, Tr);
         HypreParMatrix *Pi;
         Tr.Get(Pi);
         P.insert(P.begin(), new HypreParMatrix(*Pi));
      }

      if (pfes.GetMyRank() == 0)
      {
         out << "MG: Adding level " << P.size() << endl;
      }

      // Copy fes as next coarse fes
      delete cpfes;
      cpfes = new ParFiniteElementSpace(pfes);
   }

   ~Multigrid() {}

   const Operator *op = nullptr;
   ParFiniteElementSpace *cpfes = nullptr; // Previous coarse FEs
   std::vector<HypreParMatrix *> P;        // Prolongations
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b->Assemble();

   // 6. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   if (sigma == -1.0)
   {
      PCG(A, M, *b, x, 1, 500, 1e-12, 0.0);
   }
   else
   {
      GMRES(A, M, *b, x, 1, 500, 10, 1e-12, 0.0);
   }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
