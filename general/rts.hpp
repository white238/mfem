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
#ifndef MFEM_RTS_HPP
#define MFEM_RTS_HPP

#include "dbg.hpp"
#include "globals.hpp"

//#include <map>
#include <list>
#include <algorithm>

namespace mfem
{
class Runtime
{
private:

   //typedef std::pair<void*,void*> EdgeKey;
   //typedef std::string EdgeVal;
   //typedef std::map<EdgeKey,EdgeVal> EdgeMap;
   //EdgeMap map;
   typedef std::list<void*> void_ptr_l;
   typedef std::list<std::string> kernel_l;
   void_ptr_l in, inout, out;
   kernel_l kernels;
   static Runtime runtime_singleton;
   bool ready = false;
   bool i_am_this = false;
   Runtime(Runtime const&);
   void operator=(Runtime const&);
   static Runtime& Get() { return runtime_singleton; }
   void Setup();
   void RW_(const void *p, const bool use_dev, int m=0);
   void RW_(void *p, const bool use_dev, int m=0);
public:
   Runtime():ready(false), i_am_this(false) { Setup(); }
   ~Runtime();
   void Print(std::ostream &out = mfem::out);
   static inline bool IsReady() { return Get().ready; }

   static inline void InOutClear()
   {
      Get().in.clear();
      Get().out.clear();
      Get().inout.clear();
   }

   template <typename T> static const T *R(const T *p, const bool use_dev)
   { Get().RW_(p,use_dev,0); return p; }
   template <typename T> static T *W(T *p, const bool use_dev)
   { Get().RW_(p,use_dev,1); return p; }
   template <typename T> static T *RW(T *p, const bool use_dev)
   { Get().RW_(p,use_dev,2); return p; }
   static void Sync();
   static void Kernel(const bool use_dev,
                      const char *file, const int line,
                      const char *function, const char *s_body);
   static void Memcpy(const char *function, void *dst, const void *src,
                      size_t bytes);
};

} // mfem

#endif // MFEM_RTS_HPP
