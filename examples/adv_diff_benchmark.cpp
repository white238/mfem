#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

class Multigrid : public Solver
{
public:
   Multigrid() : Solver() {}

   virtual void Mult(const Vector &x_, Vector &y_) const
   {
      x.front() = x_;
      b.front() = x_;
      r.front() = 0.0;

      // Pre-smoothing
      for (int i = 0; i < P.size(); ++i)
      {
         // Smooth
         S[i]->Mult(b[i], x[i]);

         // r = b - A * x
         A[i]->Mult(x[i], r[i]);
         add(b[i], -1.0, r[i], r[i]);

         // Restrict to next coarse grid
         P[i]->MultTranspose(r[i], b[i + 1]);
      }

      // Coarse solve
      coarse_solver->Mult(b.back(), x.back());

      // Post-smoothing
      for (int i = nlevels - 2; i >= 0; --i)
      {
         // Prolongate to next fine grid
         P[i]->Mult(x[i + 1], x[i]);

         // r = b - A * x
         A[i]->Mult(x[i], r[i]);
         add(b[i], -1.0, r[i], r[i]);

         // Smooth
         S[i]->Mult(r[i], b[i]);
         x[i].Add(2.0 / 3.0, b[i]);
      }

      y_ = x.front();
   }

   virtual void SetOperator(const Operator &op_)
   {
      op = &op_;
      height = op->Height();
      width = op->Width();
   }

   void AddLevel(const HypreParMatrix &mat, const ParFiniteElementSpace &pfes)
   {
      A.insert(A.begin(), new HypreParMatrix(mat));
      x.insert(x.begin(), Vector(A.front()->Width()));
      b.insert(b.begin(), Vector(A.front()->Width()));
      r.insert(r.begin(), Vector(A.front()->Width()));

      if (A.size() == 1)
      {
         // Build coarse grid solver
         // Arow = new SuperLURowLocMatrix(*A[0]);
         // superlu = new SuperLUSolver(MPI_COMM_WORLD);
         // superlu->SetPrintStatistics(false);
         // superlu->SetSymmetricPattern(true);
         // superlu->SetColumnPermutation(superlu::PARMETIS);
         // superlu->SetOperator(*Arow);
         coarse_pc = new BlockILU(*const_cast<HypreParMatrix *>(A[0]),
                                  pfes.GetFE(0)->GetDof());

         coarse_solver = new GMRESSolver(MPI_COMM_WORLD);
         static_cast<GMRESSolver *>(coarse_solver)->iterative_mode = false;
         static_cast<GMRESSolver *>(coarse_solver)->SetPreconditioner(*coarse_pc);
         static_cast<GMRESSolver *>(coarse_solver)->SetOperator(*A[0]);
         static_cast<GMRESSolver *>(coarse_solver)->SetMaxIter(500);
         static_cast<GMRESSolver *>(coarse_solver)->SetRelTol(1e-8);
      }

      if (cpfes != nullptr)
      {
         OperatorHandle Tr(Operator::Hypre_ParCSR);
         pfes.GetTrueTransferOperator(*cpfes, Tr);
         HypreParMatrix *Pi;
         Tr.Get(Pi);
         P.insert(P.begin(), new HypreParMatrix(*Pi));
         S.insert(S.begin(),
                  new BlockILU(*const_cast<HypreParMatrix *>(A.front()),
                               pfes.GetFE(0)->GetDof()));
      }

      // Copy fes as next coarse fes
      delete cpfes;
      cpfes = new ParFiniteElementSpace(pfes);

      nlevels++;
   }

   ~Multigrid()
   {
      for (int i = 0; i < A.size(); ++i)
      {
         delete A[i];
      }

      for (int i = 0; i < P.size(); ++i)
      {
         delete P[i];
      }

      for (int i = 0; i < S.size(); ++i)
      {
         delete S[i];
      }

      delete coarse_solver;
      delete coarse_pc;
      // delete Arow;
   }

   const Operator *op = nullptr;

   // Operator *Arow = nullptr;

   Solver *coarse_solver = nullptr;
   Solver *coarse_pc = nullptr;

   ParFiniteElementSpace *cpfes = nullptr; // Previous coarse FEs

   std::vector<HypreParMatrix *> P; // Prolongations

   std::vector<HypreParMatrix *> A; // System matrices

   mutable std::vector<Vector> x, b, r;

   std::vector<Solver *> S; // Smoothers

   int nlevels = 0;
};

void Jn_function()
{
   double Jo = 30; // A m^-2
   double gamma = 1.0;
   double alpha0 = 0.5;
   double alpha1 = 0.5;
   double temp = 25.0; // degC
   double nc = 2.0;
   double Na = 6.02214076e23;
   double k = 1.380649e-23; // J K^-1
   double e = 1.602176634e-19;
   double R = Na * k;
   double F = Na * e;
}

// void velocity_function(const Vector &x, Vector &u)
// {
//    double yi = x(1);

//    double avg_vel = 0.03;
//    double h = 0.01;

//    u(0) = 6.0 * avg_vel / std::pow(h, 2.0) * yi * (h - yi);
//    // u(0) = 0.0;
//    u(1) = 0.0;

//    if (u.Size() == 3)
//    {
//       u(2) = 0.0;
//    }
// }

void velocity_function(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 2.0 * yi * (1.0 - xi * xi);
   u(1) = -2.0 * xi * (1.0 - yi * yi);
}

double u0_function(const Vector &x)
{
   return 0.0;
}

double inflow_function(const Vector &x)
{
   double alpha = 10.0;
   return 1.0 + tanh(alpha * (2.0 * x[0] + 1.0));
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/inline-quad.mesh";
   int ser_ref_levels = 0;
   int order = 2;
   int prob = 0;

   double d_coeff = 0.0;

   if (prob == 0)
   {
      d_coeff = 7.2e-10;
   }

   if (prob == 1)
   {
      // Cu2+
      d_coeff = 7.2e-10;
      // double d_coeff = 1.0;
      // double d_coeff = 0.0;
      // SO2-
      // double d_coeff = 10.65e-10;
      // H+
      // double d_coeff = 93.12e-10;
   }

   double d_sigma = -1.0;
   double d_kappa = -1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (d_kappa < 0)
   {
      d_kappa = (order + 1) * (order + 1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   socketstream sout;
   char vishost[] = "localhost";
   int visport = 19916;
   sout.open(vishost, visport);

   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   Mesh *mesh = new Mesh(2, 2, Element::QUADRILATERAL, false, 2.0, 1.0);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   for (int v = 0; v < mesh->GetNV(); ++v)
   {
      mesh->GetVertex(v)[0] -= 1.0;
   }

   for (int e = 0; e < mesh->GetNBE(); ++e)
   {
      Element *el = mesh->GetBdrElement(e);

      Array<int> v;
      el->GetVertices(v);

      const double *coordsa = mesh->GetVertex(v[0]);
      const double *coordsb = mesh->GetVertex(v[1]);

      if (coordsa[1] == 0.0 && coordsb[1] == 0.0)
      {
         if (coordsa[0] <= 1e-8 && coordsb[0] <= 1e-8)
         {
            // Inflow
            el->SetAttribute(2);
         }
         else
         {
            // Outflow
            el->SetAttribute(3);
         }
      }
      else
      {
         // Walls
         el->SetAttribute(1);
      }
   }

   mesh->FinalizeMesh();
   mesh->EnsureNCMesh();

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   DG_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   d_coeff *= -1.0;
   ConstantCoefficient diff_coef(d_coeff);
   ConstantCoefficient minus_diff_coef(-d_coeff);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient u0(u0_function);

   Array<int> walls_attr(pmesh->bdr_attributes.Max());
   walls_attr = 0;
   walls_attr[0] = 1;

   Array<int> inflow_attr(pmesh->bdr_attributes.Max());
   inflow_attr = 0;
   inflow_attr[1] = 1;

   Array<int> outflow_attr(pmesh->bdr_attributes.Max());
   outflow_attr = 0;
   outflow_attr[2] = 1;

   Array<int> dirichlet_attr(pmesh->bdr_attributes.Max());
   dirichlet_attr = 0;
   dirichlet_attr[0] = 1;
   dirichlet_attr[1] = 1;

   ParBilinearForm *aform = new ParBilinearForm(fes);
   aform->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   aform->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   aform->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   aform->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   aform->AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(diff_coef, d_sigma, d_kappa));
   aform->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef,
                                                         d_sigma,
                                                         d_kappa),
                               dirichlet_attr);

   ParLinearForm *b = new ParLinearForm(fes);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   //   ConstantCoefficient c_bulk_coeff(c_bulk);

   FunctionCoefficient inflow(inflow_function);
   b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow,
                                                      velocity,
                                                      -1.0,
                                                      -0.5),
                           inflow_attr);

   b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero,
                                                       diff_coef,
                                                       d_sigma,
                                                       d_kappa),
                           walls_attr);

   b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(inflow,
                                                       minus_diff_coef,
                                                       d_sigma,
                                                       d_kappa),
                           inflow_attr);

   // b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero,
   //                                                     diff_coef,
   //                                                     d_sigma,
   //                                                     d_kappa),
   //                         outflow_attr);

   // ConstantCoefficient anode_coeff(1.0);
   // b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(anode_coeff,
   //                                                     diff_coef,
   //                                                     d_sigma,
   //                                                     d_kappa),
   //                         anode_bdr_attr);

   // ConstantCoefficient cathode_coeff(1.0);
   // b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(cathode_coeff,
   //                                                     diff_coef,
   //                                                     d_sigma,
   //                                                     d_kappa),
   //                         cathode_bdr_attr);

   //   b->Assemble();

   int skip_zeros = 0;

   ParGridFunction *x = new ParGridFunction(fes);
   *x = 0.0;

   /*
    * AMR
    */

   BilinearFormIntegrator *diff_integ = new DiffusionIntegrator(one);

   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
   RT_FECollection smooth_flux_fec(order - 1, dim);
   ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*diff_integ,
                                       *x,
                                       flux_fes,
                                       smooth_flux_fes);

   double max_elem_error = 1e-8;
   int nc_limit = 1;
   double hysteresis = 0.15;

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.1); // use purely local threshold
   refiner.SetLocalErrorGoal(max_elem_error);
   refiner.PreferConformingRefinement();
   refiner.SetNCLimit(nc_limit);

   ThresholdDerefiner derefiner(estimator);
   derefiner.SetThreshold(hysteresis * max_elem_error);
   derefiner.SetNCLimit(nc_limit);

   VisItDataCollection visit_dc("echem_output", pmesh);
   visit_dc.RegisterField("Cu2+", x);

   Multigrid *pc = new Multigrid();

   const int max_dofs = 1e7;
   int maxit = 5;
   for (int it = 0; it <= maxit; it++)
   {
      HYPRE_Int global_dofs = fes->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      b->Assemble();
      aform->Assemble(skip_zeros);
      aform->Finalize(skip_zeros);

      HypreParMatrix *A = aform->ParallelAssemble();
      HypreParVector *B = b->ParallelAssemble();
      HypreParVector *X = x->GetTrueDofs();

      // BlockILU ilu(*A, fes->GetFE(0)->GetDof());

      pc->AddLevel(*A, *fes);

      GMRESSolver gmres(MPI_COMM_WORLD);
      // gmres.iterative_mode = false;
      gmres.SetPreconditioner(*pc);
      gmres.SetOperator(*A);
      gmres.SetMaxIter(100);
      gmres.SetRelTol(1e-8);
      gmres.SetPrintLevel(1);

      //*X = 0.0;
      gmres.Mult(*B, *X);

      x->Distribute(*X);

      visit_dc.SetCycle(it);
      visit_dc.Save();

      if (global_dofs > max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // pmesh->UniformRefinement();

      refiner.Apply(*pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      fes->Update();
      x->Update();

      // if (pmesh->Nonconforming())
      // {
      //    pmesh->Rebalance();
      //    fes->Update();
      //    x->Update();
      // }

      aform->Update();
      b->Update();

      // if (derefiner.Apply(*pmesh))
      // {
      //    if (myid == 0)
      //    {
      //       cout << "\nDerefined elements." << endl;
      //    }

      //    fes->Update();
      //    x->Update();

      //    if (pmesh->Nonconforming())
      //    {
      //       pmesh->Rebalance();
      //       fes->Update();
      //       x->Update();
      //    }

      //    aform->Update();
      //    b->Update();
      // }

      sout << "parallel " << num_procs << " " << myid << "\n";
      sout << "solution\n" << *pmesh << *x << flush;
      if (it == 0)
      {
         sout << "keys rRjlca" << endl;
      }

      delete A;
      delete B;
      delete X;
      // delete Arow;
      // delete superlu;
   }

   MPI_Finalize();
   return 0;
}
