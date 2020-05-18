#include "mfem.hpp"
#include "petscmat.h"

using namespace std;
using namespace mfem;

void velocity_mms(const Vector &coords, Vector &u)
{
   double x = coords(0);
   double y = coords(1);

   u(0) = cos(x) * sin(y);
   u(1) = -sin(x) * cos(y);
}

double pressure_mms(const Vector &coords)
{
   double x = coords(0);
   double y = coords(1);

   return cos(x) * sin(y);
}

void forcing_mms(const Vector &coords, Vector &u)
{
   double x = coords(0);
   double y = coords(1);

   u(0) = -cos(x) * (sin(x) - 2.0 * sin(y)) - sin(x) * sin(y);
   u(1) = cos(y) * (cos(x) - 2.0 * sin(x) - sin(y));
}

double brinkmann_func(const Vector &coords)
{
   double x = coords(0);
   double y = coords(1);

   double cx = 0.5;
   double cy = 0.5;
   double r = 0.1;

   double eps = 1e-8;

   // Circle
   if (sqrt(pow(x - cx, 2.0) + pow(y - cy, 2.0)) <= r)
   {
      return 1.0 / eps;
   }

   // Square
   // if ((x >= (cx - r) && x <= (cx + r)) && (y >= (cy - r) && y <= (cy + r)))
   // {
   //    return 1.0 / eps;
   // }

   return 0.0;
}

class NavierStokesOperator : public Operator
{
public:
   NavierStokesOperator(ParBlockNonlinearForm *form);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual Operator &GetGradient(const Vector &x) const;

   ParBlockNonlinearForm *Hform_;

   mutable PetscParMatrix *Hpetsc_ = nullptr;
};

class NavierStokesIntegrator : public BlockNonlinearFormIntegrator
{
public:
   NavierStokesIntegrator(VectorCoefficient *forcingCoeff = nullptr,
                          Coefficient *brinkmannCoeff = nullptr)
      : forcingCoeff_(forcingCoeff), brinkmannCoeff_(brinkmannCoeff)
   {}

   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec);

   virtual void AssembleElementGrad(const Array<const FiniteElement *> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats);

   // Element Jacobian inverse
   DenseMatrix Ji;
   // Shape function derivative in physical space var u
   DenseMatrix DSp_u;
   // Shape function divergence in phyiscal space var u
   Vector DivSp_u;
   // Input physical function values on the element in matrix form (dof x dim)
   DenseMatrix MatI_u;
   // Output physical function values on the element in matrix form (dof x dim)
   DenseMatrix MatO_u;
   // Shape function in physical space var u
   Vector Sp_u;
   // Shape function in physical space var p
   Vector Sp_p;
   // Shape function derivative in physical space var p
   DenseMatrix DSp_p;
   // Forcing coefficient
   VectorCoefficient *forcingCoeff_;
   // Forcing evaluated on every quadrature point of the element
   DenseMatrix Mat_f;
   // Brinkmann coefficient epsilon
   Coefficient *brinkmannCoeff_;
};

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   MFEMInitializePetsc(&argc, &argv, nullptr, nullptr);

   const char *mesh_file = "../data/inline-quad.mesh";
   int ser_ref_levels = 4;
   int order = 1;

   Mesh *mesh = new Mesh(2, 2, Element::QUADRILATERAL, false, 1.0, 1.0);
   int dim = mesh->Dimension();

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Dummy scope to force object delete
   {
      Array<int> ess_bdr_u(pmesh->bdr_attributes.Max());
      Array<int> ess_bdr_p(pmesh->bdr_attributes.Max());
      Array<Array<int> *> ess_bdr(2);
      ess_bdr_u = 1;
      ess_bdr_p = 1;

      ess_bdr[0] = &ess_bdr_u;
      ess_bdr[1] = &ess_bdr_p;

      H1_FECollection vfec(order + 1, dim);
      H1_FECollection pfec(order, dim);
      ParFiniteElementSpace vfes(pmesh, &vfec, dim);
      ParFiniteElementSpace pfes(pmesh, &pfec);

      int vdofs = vfes.GlobalTrueVSize();
      int pdofs = pfes.GlobalTrueVSize();

      if (myid == 0)
      {
         printf("#dofs (#vdofs + #pdofs) = %d (%d + %d)\n",
                vdofs + pdofs,
                vdofs,
                pdofs);
      }

      Array<ParFiniteElementSpace *> fes(2);
      fes[0] = &vfes;
      fes[1] = &pfes;

      Array<int> block_trueOffsets(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = vfes.TrueVSize();
      block_trueOffsets[2] = pfes.TrueVSize();
      block_trueOffsets.PartialSum();

      BlockVector x(block_trueOffsets);
      BlockVector y(block_trueOffsets);
      x = 0.0;
      y = 0.0;

      ParGridFunction u_gf(&vfes);
      ParGridFunction p_gf(&pfes);
      u_gf = 1.23;
      p_gf = 4.56;

      VectorFunctionCoefficient u0_coeff(dim, velocity_mms);
      u_gf.ProjectBdrCoefficient(u0_coeff, ess_bdr_u);

      FunctionCoefficient p0_coeff(pressure_mms);
      p_gf.ProjectBdrCoefficient(p0_coeff, ess_bdr_p);

      VectorFunctionCoefficient f0_coeff(dim, forcing_mms);

      FunctionCoefficient brinkmann_coeff(brinkmann_func);

      u_gf.GetTrueDofs(x.GetBlock(0));
      p_gf.GetTrueDofs(x.GetBlock(1));

      ParBlockNonlinearForm Hform(fes);
      Hform.AddDomainIntegrator(
         new NavierStokesIntegrator(&f0_coeff, &brinkmann_coeff));

      Array<Vector *> empty(2);
      empty = nullptr; // Set all entries in the array
      Hform.SetEssentialBC(ess_bdr, empty);

      NavierStokesOperator navop(&Hform);

      PetscNonlinearSolver nlsolver(MPI_COMM_WORLD);
      nlsolver.iterative_mode = true;
      nlsolver.SetOperator(navop);

      Vector zero;
      nlsolver.Mult(zero, x);

      u_gf.SetFromTrueDofs(x.GetBlock(0));
      p_gf.SetFromTrueDofs(x.GetBlock(1));

      int order_quad = 4 * order;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double err_u = u_gf.ComputeL2Error(u0_coeff, irs);
      double norm_u = ComputeGlobalLpNorm(2, u0_coeff, *pmesh, irs);
      double err_p = p_gf.ComputeL2Error(p0_coeff, irs);
      double norm_p = ComputeGlobalLpNorm(2, p0_coeff, *pmesh, irs);

      printf("|| u_h - u_ex || / || u_ex || = %.2E\n", err_u / norm_u);
      printf("|| p_h - p_ex || / || p_ex || = %.2E\n", err_p / norm_p);

      char vishost[] = "localhost";
      int visport = 19916;

      socketstream solu_sock(vishost, visport);
      solu_sock << "parallel " << num_procs << " " << myid << "\n";
      solu_sock.precision(8);
      solu_sock << "solution\n" << *pmesh << u_gf << "keys rRljc\n" << flush;

      socketstream solp_sock(vishost, visport);
      solp_sock << "parallel " << num_procs << " " << myid << "\n";
      solp_sock.precision(8);
      solp_sock << "solution\n" << *pmesh << p_gf << "keys rRljc\n" << flush;
   }

   delete pmesh;

   MFEMFinalizePetsc();
   MPI_Finalize();

   return 0;
}

void NavierStokesIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   MFEM_ASSERT(el.Size() == 2, "Expecting exactly two FE spaces");

   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   double Wq = 0.0;

   elvec[0]->SetSize(dof_u * dim);
   elvec[1]->SetSize(dof_p);

   Ji.SetSize(dim);
   Sp_u.SetSize(dof_u);
   DSp_u.SetSize(dof_u, dim);
   DivSp_u.SetSize(dof_u * dim);
   MatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   MatO_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);
   const Vector &VecI_p = *elfun[1];
   Vector &VecO_p = *elvec[1];
   DenseMatrix G_u(dim);

   DSp_p.SetSize(dof_p, dim);
   Sp_p.SetSize(dof_p);
   Vector G_p(dim);

   int intorder = 3 * el[0]->GetOrder() + 3;
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   if (forcingCoeff_)
   {
      // Evaluate forcing on every quadrature point of the element
      forcingCoeff_->Eval(Mat_f, Tr, ir); // (dim x qp)
   }

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;
   MatO_u = 0.0;
   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);

      el[0]->CalcPhysShape(Tr, Sp_u);   // (dof_u)
      el[0]->CalcPhysDShape(Tr, DSp_u); // (dof_u x dim)
      MultAtB(MatI_u, DSp_u, G_u);      // (dim x dim)

      el[1]->CalcPhysShape(Tr, Sp_p);   // (dof_p)
      el[1]->CalcPhysDShape(Tr, DSp_p); // (dof_p x dim)
      DSp_p.MultTranspose(VecI_p, G_p); // (dim)

      double pval = Sp_p * VecI_p;

      Vector uval(3);
      uval = 0.0;
      for (int i = 0; i < dof_u; ++i)
      {
         for (int c = 0; c < dim; ++c)
         {
            uval(c) += Sp_u(i) * MatI_u(i, c);
         }
      }

      Wq = ip.weight * Tr.Weight();

      for (int i = 0; i < dof_u; ++i)
      {
         // Components
         for (int c = 0; c < dim; ++c)
         {
            // Derivatives
            for (int d = 0; d < dim; ++d)
            {
               // dot(u, grad(u)) * v
               MatO_u(i, c) += uval(d) * G_u(c, d) * Sp_u(i) * Wq;

               // grad(u) : grad(v)
               MatO_u(i, c) += G_u(c, d) * DSp_u(i, d) * Wq;
            }
            // -div(v) * p
            MatO_u(i, c) -= DSp_u(i, c) * pval * Wq;
         }
      }

      double div_u = G_u.Trace();
      for (int i = 0; i < dof_p; ++i)
      {
         // -q * div(u)
         VecO_p(i) -= Sp_p(i) * div_u * Wq;
      }

      double brinkmannCoeffEval = 0.0;
      if (brinkmannCoeff_)
      {
         brinkmannCoeffEval = brinkmannCoeff_->Eval(Tr, ip);
      }

      for (int i = 0; i < dof_u; ++i)
      {
         // Component
         for (int c = 0; c < dim; ++c)
         {
            if (forcingCoeff_)
            {
               // -f_c * v_i
               MatO_u(i, c) -= Mat_f(c, q) * Sp_u(i) * Wq;
            }

            if (brinkmannCoeff_)
            {
               // brinkmannCoeff * u_c * v_i
               MatO_u(i, c) += brinkmannCoeffEval * uval(c) * Sp_u(i) * Wq;
            }
         }
      }
   }
}

void NavierStokesIntegrator::AssembleElementGrad(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   double Wq = 0.0;

   elmats(0, 0)->SetSize(dof_u * dim, dof_u * dim);
   elmats(0, 1)->SetSize(dof_u * dim, dof_p);
   elmats(1, 0)->SetSize(dof_p, dof_u * dim);
   elmats(1, 1)->SetSize(dof_p, dof_p);

   *elmats(0, 0) = 0.0;
   *elmats(0, 1) = 0.0;
   *elmats(1, 0) = 0.0;
   *elmats(1, 1) = 0.0;

   Ji.SetSize(dim);
   Sp_u.SetSize(dof_u);
   DSp_u.SetSize(dof_u, dim);
   MatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   DenseMatrix G_u(dim);

   Sp_p.SetSize(dof_p);

   int intorder = 3 * el[0]->GetOrder() + 3;
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);

      el[0]->CalcPhysShape(Tr, Sp_u);   // (dof_u)
      el[0]->CalcPhysDShape(Tr, DSp_u); // (dof_u x dim)
      MultAtB(MatI_u, DSp_u, G_u);      // (dim x dim)

      el[1]->CalcPhysShape(Tr, Sp_p);   // (dof_p)
      el[1]->CalcPhysDShape(Tr, DSp_p); // (dof_p x dim)

      Wq = ip.weight * Tr.Weight();

      Vector uval(3);
      uval = 0.0;
      for (int i = 0; i < dof_u; ++i)
      {
         for (int c = 0; c < dim; ++c)
         {
            uval(c) += Sp_u(i) * MatI_u(i, c);
         }
      }

      double brinkmannCoeffEval = 0.0;
      if (brinkmannCoeff_)
      {
         brinkmannCoeffEval = brinkmannCoeff_->Eval(Tr, ip);
      }

      // [ uv, pv ]
      // [ qu, 0  ]

      // u, v
      for (int i = 0; i < dof_u; i++)
      {
         for (int j = 0; j < dof_u; j++)
         {
            for (int c = 0; c < dim; c++)
            {
               for (int d = 0; d < dim; d++)
               {
                  // grad(u) * v_i * v_j
                  (*elmats(0, 0))(dof_u * c + i, dof_u * d + j) += G_u(c, d)
                                                                   * Sp_u(i)
                                                                   * Sp_u(j)
                                                                   * Wq;

                  // Only diagonal contributions

                  // u * grad(v_j) * v_i
                  (*elmats(0, 0))(dof_u * c + i, dof_u * c + j) += uval(d)
                                                                   * DSp_u(j, d)
                                                                   * Sp_u(i)
                                                                   * Wq;

                  // grad(v_i) : grad(v_j)
                  (*elmats(0, 0))(dof_u * c + i, dof_u * c + j) += DSp_u(i, d)
                                                                   * DSp_u(j, d)
                                                                   * Wq;
               }

               if (brinkmannCoeff_)
               {
                  // brinkmannCoeff * v_i * v_j
                  (*elmats(0, 0))(dof_u * c + i, dof_u * c + j)
                     += brinkmannCoeffEval * Sp_u(i) * Sp_u(j) * Wq;
               }
            }
         }
      }

      // p, v
      for (int i = 0; i < dof_u; i++)
      {
         for (int j = 0; j < dof_p; j++)
         {
            for (int c = 0; c < dim; c++)
            {
               // -p * div(v)
               (*elmats(0, 1))(dof_u * c + i, j) -= Sp_p(j) * DSp_u(i, c) * Wq;
            }
         }
      }

      // q, u
      // elmat10 = elmat01^T
      // (*elmats(1, 0)).Transpose(*elmats(0, 1));
      for (int i = 0; i < dof_p; i++)
      {
         for (int j = 0; j < dof_u; j++)
         {
            for (int c = 0; c < dim; c++)
            {
               // -q * div(u)
               (*elmats(1, 0))(i, dof_u * c + j) -= Sp_p(i) * DSp_u(j, c) * Wq;
            }
         }
      }
   }
}

NavierStokesOperator::NavierStokesOperator(ParBlockNonlinearForm *form)
   : Operator(), Hform_(form)
{
   this->height = form->Height();
   this->width = form->Width();
}

void NavierStokesOperator::Mult(const Vector &x, Vector &y) const
{
   Hform_->Mult(x, y);
}

Operator &NavierStokesOperator::GetGradient(const Vector &x) const
{
   Hform_->SetGradientType(Operator::PETSC_MATAIJ);

   BlockOperator &H = Hform_->GetGradient(x);

   delete Hpetsc_;
   Hpetsc_ = new PetscParMatrix(MPI_COMM_WORLD, &H, Operator::PETSC_MATAIJ);

   return *Hpetsc_;

   return H;
}
