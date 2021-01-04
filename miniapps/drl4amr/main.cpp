#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "drl4amr.hpp"

int main(int argc, char *argv[])
{
   const int N = argc > 1 ? atoi(argv[1]) : 16;
   const int order = 3;
   Drl4Amr sim(order);

   for (int i = 0; (i<N) && sim.GetNorm() > 0.01; i++)
   {
      const int e = static_cast<int>(drand48()*sim.GetNE());
      sim.Compute();
      sim.Refine(e);
      sim.GetImage();
      sim.GetIdField();
      sim.GetDepthField();
   }
   //sim.Save("m");
   return 0;
}
