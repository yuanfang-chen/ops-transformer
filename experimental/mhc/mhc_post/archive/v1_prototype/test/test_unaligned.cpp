/*
 * MIT License
 *
 * Copyright (c) 2025
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"

extern "C" void mhc_post_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t, int64_t);

int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream; aclrtCreateStream(&stream);
    
    int64_t batch=2, seq_len=4, dim=7, num_streams=4;  // dim=7 不对齐
    int64_t in_sz = batch*seq_len*dim, out_sz = batch*num_streams*seq_len*dim;
    
    std::vector<float> h_in(in_sz), h_hp(num_streams), h_out(out_sz), h_ref(out_sz);
    for(int i=0;i<in_sz;i++) h_in[i] = (i%10)*0.1f;
    float sum=0; for(int i=0;i<num_streams;i++) { h_hp[i]=i+1; sum+=h_hp[i]; }
    for(int i=0;i<num_streams;i++) h_hp[i]/=sum;
    
    for(int b=0;b<batch;b++) for(int s=0;s<num_streams;s++) {
        float w=h_hp[s]; int ob=b*num_streams+s;
        for(int i=0;i<seq_len*dim;i++) h_ref[ob*seq_len*dim+i]=h_in[b*seq_len*dim+i]*w;
    }
    
    void *d_in,*d_hp,*d_out;
    aclrtMalloc(&d_in,in_sz*4,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_hp,num_streams*4,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_out,out_sz*4,ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_in,in_sz*4,h_in.data(),in_sz*4,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_hp,num_streams*4,h_hp.data(),num_streams*4,ACL_MEMCPY_HOST_TO_DEVICE);
    
    mhc_post_do(batch*num_streams,stream,(uint8_t*)d_in,(uint8_t*)d_hp,(uint8_t*)d_out,batch,seq_len,dim,num_streams);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_out.data(),out_sz*4,d_out,out_sz*4,ACL_MEMCPY_DEVICE_TO_HOST);
    
    float maxdiff=0;
    for(int i=0;i<out_sz;i++) maxdiff=std::max(maxdiff,std::abs(h_out[i]-h_ref[i]));
    std::cout << "dim=7 (unaligned) max_diff=" << maxdiff << " -> " << (maxdiff<1e-5?"PASS":"FAIL") << std::endl;
    
    aclrtFree(d_in);aclrtFree(d_hp);aclrtFree(d_out);
    aclrtDestroyStream(stream);aclrtResetDevice(0);aclFinalize();
    return maxdiff<1e-5?0:1;
}
