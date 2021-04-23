#ifndef __LOR_AMS_HPP__
#define __LOR_AMS_HPP__

#include "mfem.hpp"

namespace mfem
{

class LOR_AMS : public Solver
{
private:
   ParFiniteElementSpace &nd_fes;
   GeneralAMS *ams;
   HypreParMatrix *G, *Pi, *B_G, *B_Pi;
   HypreSmoother *smoother;
   HypreBoomerAMG *G_solv, *Pi_solv;
public:
   LOR_AMS(ParFiniteElementSpace &nd_fes_)
   : Solver(nd_fes_.GetTrueVSize()), nd_fes(nd_fes_) { }

   void Init(HypreParMatrix &A)
   {
      ParMesh &mesh = *nd_fes.GetParMesh();
      int dim = mesh.Dimension();
      H1_FECollection h1_fec(1, dim);
      ParFiniteElementSpace h1_fes(&mesh, &h1_fec);
      ParFiniteElementSpace h1_vec_fes(&mesh, &h1_fec, dim, Ordering::byVDIM);

      ParDiscreteLinearOperator grad(&h1_fes, &nd_fes);
      grad.AddDomainInterpolator(new GradientInterpolator);
      grad.Assemble();
      grad.Finalize();
      G = grad.ParallelAssemble();
      B_G = RAP(&A, G);
      G_solv = new HypreBoomerAMG(*B_G);

      // H1 and ND spaces are both lowest order: should be simpler way to
      // construct the interpolation?
      ParDiscreteLinearOperator interp(&h1_vec_fes, &nd_fes);
      interp.AddDomainInterpolator(new IdentityInterpolator);
      interp.Assemble();
      interp.Finalize();
      Pi = interp.ParallelAssemble();
      B_Pi = RAP(&A, Pi);
      Pi_solv = new HypreBoomerAMG(*B_Pi);
      Pi_solv->SetSystemsOptions(dim);

      smoother = new HypreSmoother(A, 1); // Best smoother options?

      ams = new GeneralAMS(A, *Pi, *G, *Pi_solv, *G_solv, *smoother);
   }

   void Mult(const Vector &x, Vector &y) const
   {
      ams->Mult(x, y);
   }

   void SetOperator(const Operator &op)
   {
      const HypreParMatrix *A = dynamic_cast<const HypreParMatrix*>(&op);
      MFEM_VERIFY(A != NULL, "");
      Init(const_cast<HypreParMatrix&>(*A));
   }

   ~LOR_AMS()
   {
      delete G;
      delete B_G;
      delete G_solv;
      delete Pi;
      delete B_Pi;
      delete Pi_solv;
      delete ams;
   }
};

} // namespace mfem

#endif
