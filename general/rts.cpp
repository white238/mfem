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

#include "rts.hpp"
#include "forall.hpp"

#include <functional>
#include <list>
#include <string>
#include <unordered_set>
#include <fstream>

namespace mfem
{

// Initialize the unique global Runtime variable.
Runtime Runtime::runtime_singleton;

Runtime::~Runtime()
{
   if (i_am_this) { /* only *this will do something here */}
}

void Runtime::Setup()
{
   InOutClear();
   ilk = 0;
   known_kernels.clear();
   input_address.clear();
   output_address.clear();
   known_kernels.clear();
   loop_kernels.clear();
   // Copy all data members from the global 'runtime_singleton' into '*this'.
   std::memcpy(this, &Get(), sizeof(Runtime));
   i_am_this = true;
}

void Runtime::Start_()
{
   if (ready) {return;}
   MFEM_VERIFY(!ready && !record,"");
   dbg("\033[7;1;37mRuntime::Start");
   record = true;
   loop_kernels.push_back({-1,"",0,"","","<<<Start>>>"});
}

void Runtime::DumpGraph_()
{
   dbg("\033[31;7;1mDumpGraph_");
   std::ofstream graph_of;
   graph_of.open ("kernels.dot");
   graph_of << "digraph {\nordering=out;" << std::endl;
   graph_of << "rankdir=\"TB\"" << std::endl;
   graph_of << "edge[arrowhead=open];" << std::endl;
   graph_of << "graph [fontname = \"helvetica\"];" << std::endl;
   graph_of << "node [fontname = \"helvetica\"];" << std::endl;
   graph_of << "edge [fontname = \"helvetica\"];" << std::endl;
   graph_of << "node [style = filled, shape = circle];\n" << std::endl;
   kernel_l::const_iterator k;
   // Kernels
   //kernel_l all_kernels = loop_kernels;
   /*dbg("all_kernels:");
   for (auto k : all_kernels)
   {
      if (k.n < 0) { continue; } std::cout << k.n<<":"<<k.hash << "\n";
   }*/
   for (k=loop_kernels.begin(); k != loop_kernels.end(); ++k)
   {
      if (k->n < 0) { continue; }
      graph_of << "kernel_" << k->n <<" [label=\"";
      graph_of << k->hash;
      graph_of << "\" color=\"#CCDDCC\"]" <<  std::endl;
   }
   // In/Out
   graph_of <<  std::endl;
   inputs_l::const_iterator a;
   inputs_l in = input_address;
   inputs_l out = output_address;
   inputs_l all = in;
   all.merge(out);
   all.sort();
   all.unique();
   for (a=all.begin(); a != all.end(); ++a)
   {
      graph_of << "address_" << a->address << " [label=<<B>";
      graph_of << a->address;
      graph_of << "</B>> color=\"#DD8888\"]" <<  std::endl;
   }
   // Inputs
   graph_of <<  std::endl;
   dbg("input_address:");
   for (auto v : input_address) { std::cout << v.address << "\n"; }
   for (a=input_address.begin(); a != input_address.end(); ++a)
   {
      const int n = a->kernel.n;
      dbg("Inputs n=%d",n);
      graph_of << "address_" << a->address << " -> " << "kernel_" << n << std::endl;
   }
   // Outputs
   graph_of <<  std::endl;

   dbg("output_address:");
   for (auto v : output_address) { std::cout << v.address << "\n"; }
   for (a=output_address.begin(); a != output_address.end(); ++a)
   {
      const int n = a->kernel.n;
      dbg("Outputs n=%d",n);
      graph_of << "kernel_" << n <<  " -> " << "address_" << a->address << std::endl;
   }
   dbg("done");
   graph_of << "}" << std::endl;
   graph_of.close();
   exit(0);
}


void Runtime::For_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::For");
   loop_kernels.push_back({-1,"",0,"","","<<<For>>>"});
}

void Runtime::Loop_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Loop");
   record = false; // end of recording
   ready = true; // loop should be ready
   loop_kernels.push_back({-1,"",0,"","","<<<Loop>>>"});
   DumpGraph_();
}

void Runtime::Return_()
{
   dbg("");
   loop_kernels.push_back({-1,"",0,"","","<<<Return>>>"});
}

void Runtime::Break_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Break");
   MFEM_VERIFY(!ready && !record,"");
   loop_kernels.push_back({-1,"",0,"","","<<<Break>>>"});
}

void Runtime::Cond_(const char *test)
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Cond: %s",test);
   MFEM_VERIFY(record,"");
   std::string cond("<<<Cond>>>(");
   cond.reserve(1024);
   cond += test;
   cond += ")";
   loop_kernels.push_back({-1,"",0,"","", strdup(cond.c_str())});
}

void Runtime::Stop_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Stop");
   loop_kernels.push_back({-1,"",0,"","","<<<Stop>>>"});
}


void Runtime::Print(std::ostream &out) { dbg(""); }

void Runtime::RW_(void *p, const bool use_dev, int m)
{
   if (!use_dev) { return; }
   if (!Device::IsEnabled()) { return; }
   switch (m)
   {
      case 0:
      {
         Get().in.push_back({p});
         //printf("\n\033[32m%p:%d \033[m", p, m);
         break;
      }
      case 1:
      {
         Get().out.push_back({p});
         //printf("\n\033[35m%p:%d \033[m", p, m);
         //mfem_backtrace();
         break;
      }
      case 2:
      {
         Get().inout.push_back({p});
         //printf("\n\033[31m%p:%d \033[m", p, m);
         break;
      }
      default: mfem_error();
   }
   //fflush(0);
}

void Runtime::RW_(const void *p, const bool use_dev, int m)
{
   return RW_(const_cast<void*>(p), use_dev, m);
}


void Runtime::Sync() { printf("\033[33mS\033[m"); fflush(0); }

void Runtime::Kernel(const bool use_dev,
                     const char *file, const int line,
                     const char *function, const char *s_body)
{
   if (!use_dev) { InOutClear(); return; }
   if (!Device::IsEnabled()) { InOutClear(); return; }
   //printf("\n\033[33m[kernel] '%s' (%s:%d)\033[m", function, file, line);
   //printf("\n\033[33m[kernel] %s\033[m", s_body);

   address_l &in = Get().in;
   address_l &out = Get().out;
   address_l &inout = Get().inout;
   kernel_l &kkernels = Get().known_kernels;
   kernel_l &lkernels = Get().loop_kernels;

   // Add kernel if we haven't seen him yet
   std::stringstream kernal_ss;
   kernal_ss << Get().ilk << "_" << file << ":" << line << ":" << function;
   std::string kernel_s = kernal_ss.str();
   const char *hash = kernel_s.c_str();
   auto it = std::find(kkernels.begin(), kkernels.end(), hash);
   if (it == kkernels.end())
   {
      //printf("\n\033[33m[kernel] New kernel (\033[1;33m%s)\033[m", hash);
      kernel_t k{-1,file, line, function, s_body, strdup(hash)};
      kkernels.push_back(k);
      if (Get().record)
      {
         lkernels.push_back(k);
         lkernels.back().n = Get().ilk;
      }
   }
   else
   {
      //printf("\n\033[33m[kernel] Known kernel (%s)\033[m", hash);
      if (Get().record)
      {
         lkernels.push_back(*it);
         lkernels.back().n = Get().ilk;
      }
   }

   if (Get().record)
   {
      address_l::const_iterator i;
      const kernel_t &k = lkernels.back();

      for (i=in.begin(); i != in.end(); ++i)
      {
         address_t a{*i, k};
         //printf("\n\t\033[33m[kernel] \033[32m%p\033[m", *i);
         Get().input_address.push_back(a);
      }

      for (i=out.begin(); i != out.end(); ++i)
      {
         address_t a{*i, k};
         //printf("\n\t\033[33m[kernel] \033[35m%p\033[m", *i);
         Get().output_address.push_back(a);
      }


      for (i=inout.begin(); i != inout.end(); ++i)
      {
         address_t a{*i, k};
         //printf("\n\t\033[33m[kernel] \033[31m%p\033[m", *i);
         Get().input_address.push_back(a);
         Get().output_address.push_back(a);
      }
   }
   if (Get().record) { Get().ilk++; }
   InOutClear();
}

void Runtime::Memcpy(const char *function, void *dst, const void *src,
                     size_t bytes)
{
   /*printf("\n\033[33m[%s] \033[32m%p\033[m => \033[35m%p\033[m (%ld bytes)\033[m",
          function, src, dst, bytes);*/
   if (Get().record)
   {
      std::stringstream kernal_ss;
      kernal_ss << Get().ilk << "_" << function;
      std::string kernel_s = kernal_ss.str();
      const char *hash = kernel_s.c_str();
      const kernel_t k{Get().ilk++,"", 0, "", "", strdup(hash)};
      Get().loop_kernels.push_back(k);
      const kernel_t &back = Get().loop_kernels.back();
      Get().input_address.push_back(address_t{src, back});
      Get().output_address.push_back(address_t{dst, back});
   }
   InOutClear();
}

} // mfem
